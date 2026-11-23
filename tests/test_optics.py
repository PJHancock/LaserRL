import json
import math
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
from gymnasium.utils.env_checker import check_env
from optics_env import ContinuousOpticsEnv, Puzzle, puzzle_named
from optics_experiment import prepare, load_splits, rollout, summarize
from optics_render import save_playback


class OpticsTests(unittest.TestCase):
    def test_gym_api(self):
        check_env(ContinuousOpticsEnv(),skip_render_check=True)

    def test_solution_witnesses_and_circle_hit(self):
        for name in ('single','relay'):
            env=ContinuousOpticsEnv(puzzle_named(name))
            trace=env.trace_beam(env.validate_poses(env.puzzle.solution))
            self.assertTrue(trace['solved'])
            self.assertEqual(trace['stop'],'target')
            end=np.array(trace['segments'][-1][1])
            self.assertAlmostEqual(np.linalg.norm(end-env.puzzle.target),env.puzzle.target_radius)
            self.assertEqual(len(trace['hits']),env.num_mirrors)

    def test_obstacle_blocks_beam_before_target(self):
        puzzle=Puzzle(source=(.08,.5),target=(.8,.5),obstacles=((.3,.4,.1,.2),),
                      solution=((.6,.8,0.0),))
        env=ContinuousOpticsEnv(puzzle)
        trace=env.trace_beam(puzzle.solution)
        self.assertFalse(trace['solved']);self.assertEqual(trace['stop'],'obstacle')
        self.assertAlmostEqual(trace['segments'][0][1][0],.3)

    def test_reflection_limit_and_relay_observation(self):
        env=ContinuousOpticsEnv(puzzle_named('relay'),max_reflections=1)
        trace=env.trace_beam(env.puzzle.solution)
        self.assertFalse(trace['solved'])
        self.assertEqual(trace['stop'],'reflection_limit')
        self.assertEqual(len(trace['hits']),1)
        obs,_=env.reset(seed=1)
        self.assertTrue(env.observation_space.contains(obs))

    def test_reproducible_resets_and_observations(self):
        a,b=ContinuousOpticsEnv(),ContinuousOpticsEnv()
        obs,_=a.reset(seed=4);obs2,_=b.reset(seed=4)
        np.testing.assert_equal(obs,obs2)
        self.assertTrue(a.observation_space.contains(obs))
        rng=np.random.default_rng(55)
        for _ in range(25):
            action=rng.uniform(-1,1,a.action_space.shape)
            x=a.step(action);y=b.step(action)
            np.testing.assert_equal(x[0],y[0]);self.assertEqual(x[1:4],y[1:4])
            self.assertTrue(a.observation_space.contains(x[0]))
            a.validate_poses(a.mirror_poses)
            if x[2] or x[3]:
                break

    def test_solved_transition_and_terminal_potential(self):
        env=ContinuousOpticsEnv()
        start=np.array(env.puzzle.solution)
        start[0,2]+=.2
        env.reset(options={'mirror_poses':start})
        distance=env.trace['distance']
        action=np.zeros(3);action[2]=-.2/env.max_rotation
        _,reward,terminated,truncated,info=env.step(action)
        self.assertTrue(terminated);self.assertFalse(truncated)
        self.assertAlmostEqual(reward,env.success_reward-env.step_cost+distance)
        with self.assertRaises(RuntimeError):
            env.step(action)

    def test_truncation_retains_nonterminal_potential(self):
        env=ContinuousOpticsEnv(max_steps=1)
        env.reset(seed=9);distance=env.trace['distance']
        _,reward,term,trunc,_=env.step(np.zeros(3))
        self.assertFalse(term);self.assertTrue(trunc)
        self.assertAlmostEqual(reward,-env.step_cost+(1-env.gamma)*distance)

    def test_illegal_inputs_and_periodic_orientation(self):
        env=ContinuousOpticsEnv();env.reset(seed=1)
        with self.assertRaises(ValueError):
            env.step([np.nan,0,0])
        with self.assertRaises(ValueError):
            env.reset(options={'mirror_poses':[[0,0,0]]})
        p=np.array(env.puzzle.solution)
        p[0,2]+=math.pi
        np.testing.assert_allclose(env.validate_poses(p),env.puzzle.solution)

    def test_banks_are_disjoint_and_replayable(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'splits.json'
            prepare(SimpleNamespace(puzzle='single',seed=2,train_count=4,validation_count=3,test_count=3,output=str(output)))
            puzzle,data=load_splits(output)
            env=ContinuousOpticsEnv(puzzle,start_poses=data['train'][:1])
            first,_=env.reset(seed=0);second,_=env.reset(seed=99)
            np.testing.assert_equal(first,second)
            rows,frames=rollout(ContinuousOpticsEnv(puzzle,max_steps=2),data['test'],record=True)
            rows2,_=rollout(ContinuousOpticsEnv(puzzle,max_steps=2),data['test'])
            self.assertEqual(rows,rows2)
            self.assertEqual(summarize(rows)['episodes'],3)
            html=save_playback(Path(directory)/'playback.html',puzzle,frames[0])
            self.assertIn('const frames=',html.read_text())
            data['test'][0]=data['train'][0]
            output.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_splits(output)

if __name__=='__main__':
    unittest.main()
