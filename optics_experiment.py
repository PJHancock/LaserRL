"""Reproducible PPO starting-pose diversity experiments.

Use --help, or see README.md for the full prepare/train/evaluate/render workflow.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import importlib.metadata

import numpy as np
from optics_env import ContinuousOpticsEnv, Puzzle, puzzle_named
from optics_render import save_playback


def write_json(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def csv_rows(path, rows):
    if not rows:
        return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def load_splits(path):
    data=json.loads(Path(path).read_text())
    p=data['puzzle']
    puzzle=Puzzle(**{k:tuple(tuple(r) for r in v) if k in ('solution','obstacles') else tuple(v) if k in ('source','target') else v for k,v in p.items()})
    if data['puzzle_fingerprint'] != puzzle.fingerprint:
        raise ValueError('Puzzle fingerprint mismatch')
    seen=set()
    env=ContinuousOpticsEnv(puzzle)
    for split in ('train','validation','test'):
        if not data[split]:
            raise ValueError('Each split must contain starts')
        for pose in data[split]:
            arr=env.validate_poses(pose)
            key=tuple(np.round(arr,10).ravel())
            if key in seen:
                raise ValueError('Duplicate pose across or within splits')
            seen.add(key)
            if env.trace_beam(arr)['solved']:
                raise ValueError('Split contains a solved start')
    return puzzle,data


def split_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    if min(args.train_count,args.validation_count,args.test_count)<1:
        raise ValueError('All split sizes must be positive')
    if Path(args.output).exists():
        raise ValueError('Split file already exists; choose a new output to preserve experiments')
    puzzle=puzzle_named(args.puzzle)
    env=ContinuousOpticsEnv(puzzle)
    if not env.trace_beam(env.validate_poses(puzzle.solution))['solved']:
        raise ValueError('Puzzle solution witness does not illuminate target')
    rng=np.random.default_rng(args.seed)
    poses=[]; seen=set()
    while len(poses)<args.train_count+args.validation_count+args.test_count:
        pose=env.sample_poses(rng)
        key=tuple(np.round(pose,10).ravel())
        if key not in seen:
            seen.add(key);poses.append(pose.tolist())
    n,v=args.train_count,args.validation_count
    data={'version':1,'seed':args.seed,'puzzle':asdict(puzzle),'puzzle_fingerprint':puzzle.fingerprint,
          'distribution':'uniform legal unsolved joint poses; finite disjoint banks',
          'train':poses[:n],'validation':poses[n:n+v],'test':poses[n+v:]}
    write_json(args.output,data)
    print(f'Saved {n} train / {v} validation / {args.test_count} test starts to {args.output}')


def rollout(env, poses, model=None, action_seed=90000, record=False):
    """Frozen deterministic PPO, or uniform random actions, on exact supplied poses."""
    rows=[]; frames=[]
    for i,pose in enumerate(poses):
        obs,info=env.reset(seed=10000+i,options={'mirror_poses':pose})
        rng=np.random.default_rng(action_seed+i)
        episode_frames=[env.frame()] if record else []
        total=0.0; invalid=0
        while True:
            action=rng.uniform(-1,1,env.action_space.shape).astype(np.float32) if model is None else model.predict(obs,deterministic=True)[0]
            obs,reward,terminated,truncated,info=env.step(action)
            total+=reward;invalid+=not info['action_valid']
            if record:
                episode_frames.append(env.frame())
            if terminated or truncated:
                break
        rows.append({'episode':i,'return':total,'solved':int(terminated),'steps':env.steps,
                     'solve_steps':env.steps if terminated else '', 'miss_distance':info['miss_distance'],
                     'truncated':int(truncated),'invalid_actions':invalid})
        if record:
            frames.append(episode_frames)
    return rows,frames


def summarize(rows):
    successes=[r['steps'] for r in rows if r['solved']]
    return {'episodes':len(rows),'solve_rate':float(np.mean([r['solved'] for r in rows])),
            'mean_return':float(np.mean([r['return'] for r in rows])),
            'mean_episode_steps':float(np.mean([r['steps'] for r in rows])),
            'mean_solve_steps':float(np.mean(successes)) if successes else None,
            'mean_miss_distance':float(np.mean([r['miss_distance'] for r in rows]))}


def train(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    puzzle,data=load_splits(args.splits)
    sizes=args.sizes.split(',');seeds=[int(s) for s in args.seeds.split(',')]
    if len(sizes)!=len(set(sizes)) or len(seeds)!=len(set(seeds)):
        raise ValueError('Sizes and seeds must be distinct')
    for size in sizes:
        if size!='uniform' and (int(size)<1 or int(size)>len(data['train'])):
            raise ValueError('Finite diversity must be between 1 and the prepared training bank size')
    if args.rollout_steps<2 or args.timesteps<1 or args.timesteps%args.rollout_steps or args.batch_size<2 or args.rollout_steps%args.batch_size:
        raise ValueError('Use positive timesteps divisible by rollout-steps, and rollout-steps divisible by batch-size >= 2')
    if args.eval_interval<1:
        raise ValueError('eval-interval must be positive')
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    # Each run gets the exact split data alongside its saved model.
    class ValidationCallback(BaseCallback):
        def __init__(self, evaluation_env, run_dir):
            super().__init__();self.env=evaluation_env;self.run_dir=run_dir;self.best=-1;self.history=[];self.last=0
        def evaluate(self):
            rows,_=rollout(self.env,data['validation'],self.model)
            stats=summarize(rows)
            improved=stats['solve_rate']>self.best
            self.history.append({'environment_steps':self.num_timesteps,**stats,'best_so_far':improved})
            csv_rows(self.run_dir/'validation.csv',self.history)
            if improved:
                self.best=stats['solve_rate'];self.model.save(self.run_dir/'best')
                write_json(self.run_dir/'best.json',{'environment_steps':self.num_timesteps,**stats})
        def _on_training_start(self):
            self.evaluate()
        def _on_step(self):
            if self.num_timesteps-self.last>=args.eval_interval and self.num_timesteps<args.timesteps:
                self.evaluate();self.last=self.num_timesteps
            return True
        def _on_training_end(self):
            # PPO optimizes after collecting a rollout. Measure the final weights
            # after that last optimization, rather than inside its collection.
            self.evaluate()
    for size in sizes:
        bank=None if size=='uniform' else data['train'][:int(size)]
        for seed in seeds:
            run_dir=root/f'starts-{size}'/f'seed-{seed}'
            if run_dir.exists():
                raise ValueError(f'{run_dir} exists. Choose a fresh output directory.')
            run_dir.mkdir(parents=True)
            env_kwargs={'max_steps':args.horizon,'gamma':0.99,'shaping_scale':args.shaping_scale,
                        'max_move':0.05,'max_rotation':float(np.pi/12),'step_cost':0.01,
                        'success_reward':10.0,'invalid_cost':0.01,'max_reflections':32}
            config={'size':size,'seed':seed,'requested_timesteps':args.timesteps,'env':env_kwargs,
                    'split_sha256':split_digest(args.splits),'ppo':{'n_steps':args.rollout_steps,'batch_size':args.batch_size,
                    'learning_rate':3e-4,'gamma':0.99,'n_epochs':10,'device':'cpu'},'selection':'maximum validation solve rate; ties keep earliest',
                    'software':{name:importlib.metadata.version(name) for name in ['numpy','gymnasium','stable-baselines3','torch']},
                    'environment_source_sha256':hashlib.sha256(Path(__file__).with_name('optics_env.py').read_bytes()).hexdigest()}
            write_json(run_dir/'config.json',config);write_json(run_dir/'splits.json',data)
            env=Monitor(ContinuousOpticsEnv(puzzle,start_poses=bank,**env_kwargs),str(run_dir/'training.monitor.csv'),info_keywords=('solved','miss_distance'))
            eval_env=ContinuousOpticsEnv(puzzle,**env_kwargs)
            model=PPO('MlpPolicy',env,seed=seed,verbose=1,**config['ppo'])
            callback=ValidationCallback(eval_env,run_dir)
            started=time.perf_counter()
            try:
                model.learn(total_timesteps=args.timesteps,callback=callback)
                model.save(run_dir/'final')
                write_json(run_dir/'runtime.json',{'environment_steps':model.num_timesteps,
                           'seconds_including_validation':time.perf_counter()-started})
            except Exception as error:
                write_json(run_dir/'failure.json',{'error':str(error),'environment_steps':model.num_timesteps,
                           'seconds_including_validation':time.perf_counter()-started})
                raise
            finally:
                env.close();eval_env.close()
    print(f'Training complete. Run evaluate --runs {root} to measure selected policies on test starts.')


def evaluate(args):
    from stable_baselines3 import PPO
    root=Path(args.runs); summaries=[]
    for config_path in sorted(root.glob('starts-*/seed-*/config.json')):
        run=config_path.parent
        config=json.loads(config_path.read_text());puzzle,data=load_splits(run/'splits.json')
        if config.get('environment_source_sha256') != hashlib.sha256(Path(__file__).with_name('optics_env.py').read_bytes()).hexdigest():
            raise ValueError('Environment code changed since training; preserve the run and use its original dynamics')
        if split_digest(run/'splits.json')!=config['split_sha256']:
            # JSON is serialized consistently when copied during training.
            raise ValueError('Saved split data changed since training')
        output=run/'test_evaluation'
        if output.exists():
            raise ValueError(f'{output} already exists; preserve the original test evaluation')
        if not (run/'best.zip').exists():
            print(f'Skipping unfinished run {run}');continue
        model=PPO.load(run/'best.zip',device='cpu')
        env=ContinuousOpticsEnv(puzzle,**config['env'])
        train_bank=data['train'] if config['size']=='uniform' else data['train'][:int(config['size'])]
        output.mkdir()
        for policy_name,policy in [('random',None),('ppo',model)]:
            for split,bank in [('train_reference' if config['size']=='uniform' else 'train',train_bank),('validation',data['validation']),('test',data['test'])]:
                rows,_=rollout(env,bank,policy,action_seed=90000+10000*config['seed'])
                csv_rows(output/f'{policy_name}-{split}.csv',rows)
                summaries.append({'condition':config['size'],'training_seed':config['seed'],'policy':policy_name,'split':split,**summarize(rows)})
        env.close()
    if not summaries:
        raise ValueError('No trained runs found')
    csv_rows(root/'summary.csv',summaries)
    write_json(root/'summary.json',summaries)
    print(f'Saved per-episode scores and {root / "summary.csv"}. Test data did not select checkpoints.')


def render(args):
    if args.run:
        from stable_baselines3 import PPO
        run=Path(args.run);config=json.loads((run/'config.json').read_text())
        puzzle,data=load_splits(run/'splits.json')
        model=PPO.load(run/'best.zip',device='cpu')
        env=ContinuousOpticsEnv(puzzle,**config['env'])
        pose=data[args.split][args.index]
    else:
        model=None;env=ContinuousOpticsEnv(puzzle_named(args.puzzle))
        env.reset(seed=args.seed);pose=env.mirror_poses.copy()
    rows,frames=rollout(env,[pose],model,record=True)
    save_playback(args.output,env.puzzle,frames[0],title=f'{env.puzzle.name} / {"PPO" if model else "uniform random policy"}')
    write_json(Path(args.output).with_suffix('.json'),{'puzzle':asdict(env.puzzle),'metrics':rows[0],'frames':frames[0]})
    Path(args.output).with_suffix('.svg').write_text(env.render())
    print(f'Saved {args.output}; solved={rows[0]["solved"]}, steps={rows[0]["steps"]}')


def plot(args):
    import matplotlib.pyplot as plt
    root=Path(args.runs)
    rows=list(csv.DictReader((root/'summary.csv').open()))
    conditions=sorted(set(r['condition'] for r in rows),key=lambda c:float('inf') if c=='uniform' else int(c))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for policy,style in [('ppo','-o'),('random','--o')]:
        for ax,split in zip(axes,['train','test']):
            groups=[[float(r['solve_rate']) for r in rows if r['condition']==c and r['policy']==policy and (r['split']==split or split=='train' and r['split']=='train_reference')] for c in conditions]
            means=[np.mean(g) for g in groups]
            lo=[np.min(g) for g in groups];hi=[np.max(g) for g in groups]
            x=np.arange(len(conditions));ax.plot(x,means,style,label=policy)
            ax.fill_between(x,lo,hi,alpha=.15);ax.set_xticks(x,conditions)
            ax.set(title=f'{split.title()} starts',xlabel='Number of distinct training starts',ylabel='Solve rate',ylim=(-.03,1.03))
            ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(root/'solve_rates.png',dpi=180);plt.close(fig)
    print(root/'solve_rates.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for condition in conditions:
        training,validation,step_grids=[],[],[]
        for run in sorted((root/f'starts-{condition}').glob('seed-*')):
            with (run/'validation.csv').open() as src:
                checkpoints=list(csv.DictReader(src))
            steps=np.array([int(r['environment_steps']) for r in checkpoints])
            validation.append([float(r['solve_rate']) for r in checkpoints]);step_grids.append(steps)
            with (run/'training.monitor.csv').open() as src:
                episodes=list(csv.DictReader(line for line in src if not line.startswith('#')))
            ends=np.cumsum([int(r['l']) for r in episodes])
            rewards=np.array([float(r['r']) for r in episodes])
            means=[]
            for t in steps[1:]:
                n=int(np.searchsorted(ends,t,side='right'))
                means.append(float(np.mean(rewards[max(0,n-20):n])) if n else np.nan)
            training.append(means)
        if not all(np.array_equal(step_grids[0],grid) for grid in step_grids):
            raise ValueError('Runs have different evaluation checkpoints; compare equal budgets and intervals')
        for ax,values,x in [(axes[0],np.asarray(training),step_grids[0][1:]),(axes[1],np.asarray(validation),step_grids[0])]:
            ax.plot(x,np.mean(values,axis=0),label=f'{condition} starts')
            ax.fill_between(x,np.min(values,axis=0),np.max(values,axis=0),alpha=.15)
            ax.set_xlabel('Training environment steps');ax.grid(alpha=.2);ax.legend()
    axes[0].set(title='Sampled training policy',ylabel='Return: latest 20 completed episodes')
    axes[1].set(title='Frozen validation policy',ylabel='Solve rate',ylim=(-.03,1.03))
    fig.tight_layout();fig.savefig(root/'learning_curves.png',dpi=180);plt.close(fig)
    print(root/'learning_curves.png')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare',help='Save disjoint unsolved pose banks')
    p.add_argument('--puzzle',default='single');p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--train-count',type=int,default=128);p.add_argument('--validation-count',type=int,default=16);p.add_argument('--test-count',type=int,default=64)
    p.add_argument('--output',default='experiments/single-splits.json');p.set_defaults(func=prepare)
    p=sub.add_parser('train',help='Train fixed-bank diversity conditions and optionally online uniform starts')
    p.add_argument('--splits',default='experiments/single-splits.json');p.add_argument('--sizes',default='1,16,128')
    p.add_argument('--seeds',default='0,1,2');p.add_argument('--timesteps',type=int,default=131072)
    p.add_argument('--rollout-steps',type=int,default=1024);p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--eval-interval',type=int,default=16384);p.add_argument('--horizon',type=int,default=100)
    p.add_argument('--shaping-scale',type=float,default=1.0);p.add_argument('--output',default='experiments/runs');p.set_defaults(func=train)
    p=sub.add_parser('evaluate',help='Evaluate each chosen checkpoint and random baseline once on held-out poses')
    p.add_argument('--runs',default='experiments/runs');p.set_defaults(func=evaluate)
    p=sub.add_parser('render',help='Export standalone HTML playback, SVG and rollout JSON')
    p.add_argument('--run');p.add_argument('--split',choices=['train','validation','test'],default='test');p.add_argument('--index',type=int,default=0)
    p.add_argument('--puzzle',default='single');p.add_argument('--seed',type=int,default=0);p.add_argument('--output',default='artifacts/playback.html');p.set_defaults(func=render)
    p=sub.add_parser('plot',help='Plot training-start versus test solve rates across training seeds')
    p.add_argument('--runs',default='experiments/runs');p.set_defaults(func=plot)
    args=parser.parse_args(argv);args.func(args)

if __name__=='__main__':
    main()
