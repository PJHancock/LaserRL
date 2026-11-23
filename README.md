# PPO laser optics lab

A continuous Gymnasium puzzle environment and an experiment runner for:

> How does the number of distinct starting mirror poses seen during training affect PPO's ability to solve held-out poses?

The project includes a one-mirror puzzle and a two-mirror relay, opaque obstacles,
a circular target, exact ray/segment reflections, PPO training, a uniform random
baseline, pose-bank generation, recorded metrics, manual play, and standalone HTML
playback. Rendering is an independent vector optics board inspired by the reference
concept, rather than a reproduction of Astra's application.

## Setup and quick preview

From this directory:

```bash
uv sync
uv run python optics_play.py --puzzle single
```

Open **http://127.0.0.1:8765**. Choose a mirror, use the arrow controls to translate,
and Q/E or the rotation buttons to rotate. The action-size slider allows fine
adjustments. Reset with the same seed to reproduce a puzzle. This server binds only
to localhost. Try `--puzzle relay` for two mirrors.

No server is needed for saved playback: open
[`artifacts/single-demo.html`](artifacts/single-demo.html) or
[`artifacts/relay-demo.html`](artifacts/relay-demo.html) directly in a browser.
These are short **scripted geometry demonstrations**, not trained PPO results.

To export a random-agent rollout with its beam path, mirrors, and frame slider:

```bash
uv run python optics_run.py --puzzle relay --seed 7 --output artifacts/random.html
```

`main.py` is an alternative entry point for `optics_experiment.py`.

## The MDP

The board is `[0, 1] × [0, 1]`, with **y increasing downwards**.

- **Observation:** source `(x, y, cos(angle), sin(angle))`, target `(x, y, radius)`,
  each mirror's `(x, y, cos(2*angle), sin(2*angle))`, obstacle rectangles
  `(left, top, width, height)`, and remaining step fraction. Double-angle encoding
  makes equivalent mirror angles modulo π have identical observations. This vector
  includes all geometry needed to reconstruct the current beam.
- **Action:** a continuous `Box(-1, 1, shape=(3*N,))`. For every mirror, the triple
  controls `(dx, dy, dtheta)` simultaneously. A full action translates by at most
  `0.05` per axis and rotates by at most `π/12` radians (15 degrees). Positions are
  projected into the board; if any proposed mirror violates clearance, the whole
  joint action is rejected. These are discrete control steps, not swept physical
  movement along the intervening path.
- **Optics:** two-sided finite mirrors reflect specularly. The beam stops at the
  first obstacle, board edge, target-circle intersection, or reflection limit.
  Mirror centers use conservative radius `mirror_length/2` clearance around each
  other, obstacles, source and target. This keeps legal layouts unambiguous, but
  excludes some configurations whose thin mirror segments would not touch.
- **Reward:** `-0.01 + 10*solved + shaping - 0.01*invalid_action`, where
  `shaping = scale*(0.99*Phi(next) - Phi(current))`,
  `Phi = -distance_from_beam_to_target_circle`, and the terminal potential is zero.
  Distance is taken over the visible beam segments, not an infinite extension of
  the ray. Set `--shaping-scale 0` for a sparse-reward ablation. Keep PPO's gamma and
  the environment's shaping gamma matched. Discounted potential shaping can give a
  small reward even when distance is unchanged; this is not raw distance reduction.
- **Episode end:** illuminating the target terminates; the default 100-action cap
  truncates. Reset before stepping again. Gymnasium/SB3 handle time-limit
  bootstrapping separately from true termination. Resets reject already-solved
  poses so a policy cannot receive free success at action zero.

Initial poses, source, target, and geometry are expressed in continuous values.
The source/target/obstacles remain fixed within one puzzle family. Only the mirror
start bank changes in the primary experiment. Each preset includes a verified
solution witness. This establishes that its optical layout is solvable; it does
not prove every arbitrary start can reach that witness in 100 legal actions.

### Python API

```python
from optics_env import ContinuousOpticsEnv, puzzle_named

env = ContinuousOpticsEnv(puzzle_named("single"), render_mode="svg")
obs, info = env.reset(seed=0)
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
svg = env.render()                 # SVG string; no browser needed
# info: solved, miss_distance, mirror_poses, steps, ray_stop, action_valid,
#       and the shaping reward component
```

`render_mode="rgb_array"` returns a NumPy image for video/notebooks;
`"ansi"` returns a short state summary. `reset(options={"mirror_poses": poses})`
replays an exact start. An explicit bank is passed as `start_poses` with shape
`(number_of_starts, N, 3)`. Every reset samples uniformly from that bank.

## Run the diversity experiment

### 1. Generate and freeze start splits

Two ready-made split files are included in `experiments/`, each with 128 train,
16 validation and 64 test poses (generation seed 2026). To create another set:

```bash
uv run python optics_experiment.py prepare --puzzle single --seed 2026 \
  --train-count 128 --validation-count 16 --test-count 64 \
  --output experiments/my-splits.json
```

Starts are legal unsolved joint poses sampled uniformly by rejection. Files contain
exact poses, puzzle geometry, the solution witness, and a puzzle fingerprint.
The loader checks for duplicate poses and train/validation/test overlap. Generated
files are never silently overwritten. The first K entries of the training bank
form the K-start condition, so banks are **nested**. Hold the generation seed fixed
across training seeds.

Held-out means **new exact poses from the same broad distribution**, not necessarily
a spatial or angular region absent from training. This experiment measures finite
start-set diversity/generalization. It does not establish out-of-distribution
robustness. A single training pose might be unusually easy or difficult; repeat the
whole comparison with other split-generation seeds for a stronger conclusion.

### 2. Train with equal interaction budgets

```bash
uv run python optics_experiment.py train \
  --splits experiments/single-splits.json \
  --sizes 1,16,128 --seeds 0,1,2 \
  --timesteps 131072 --output experiments/runs-single
```

All nine runs use the same physics, reward, action scales, network and PPO settings.
The only condition change is how many stored starts are available. The budget must
be divisible by `--rollout-steps` (default 1024), preventing SB3's usual rollout
rounding from producing unequal budgets. Optional `--sizes 1,16,128,uniform` adds a
condition that generates a fresh legal pose at every reset.

The same validation starts are evaluated before training and every 16,384
interactions. The actor uses deterministic PPO actions for validation; weights and
training RNGs are untouched. The checkpoint with the highest validation solve rate
is saved as `best.zip`; ties keep the earlier checkpoint, including checkpoint 0.
A `final.zip` is also saved. If no checkpoint solves validation starts, the initial
checkpoint can remain best: report that failure instead of claiming a successful
learner. Long-range random starts may be difficult with beam-distance shaping,
which is flat when no mirror intercepts the beam. Start with the one-mirror puzzle
and inspect learning before increasing puzzle complexity or changing rewards.

Each run directory contains:

- `config.json`, `splits.json`: the condition and data used;
- `training.monitor.csv`: sampled training episode return, length and solve flag;
- `validation.csv`: frozen checkpoint solve rate, return, miss distance and steps;
- `best.json`: selected checkpoint interaction count and validation statistics;
- `runtime.json`: exact training transitions and elapsed time **including validation**;
- model ZIP files, or `failure.json` for a failed run.

### 3. Evaluate the selected models and random baseline

```bash
uv run python optics_experiment.py evaluate --runs experiments/runs-single
uv run python optics_experiment.py plot --runs experiments/runs-single
```

This evaluates the selected PPO checkpoint and uniform random actions on training,
validation and test banks, with one rollout per exact start. Test poses never select
weights. Random actions use a separate seeded generator; the rollout cap is the
same for both policies. The tool refuses to overwrite an existing test evaluation,
so keep the first test results and avoid using them to tune training.

Per-episode CSVs include return, solve flag, action count, miss distance, truncation
and invalid-action count. `summary.csv`/`summary.json` aggregate solve rates and
steps. **Mean solve steps is conditional on success** and is null if no episode
succeeds; mean episode steps includes failures at the cap. Always show solve rate
alongside conditional efficiency. For the online-uniform condition, the training
bank is labeled `train_reference` because those exact starts were not the finite
set used for fitting.

`solve_rates.png` compares PPO and random actions for train/test starts, showing the
mean and min-to-max range across training seeds. `learning_curves.png` shows sampled
training return (latest 20 completed episodes) and frozen validation solve rate
against training environment steps, again with mean and min-to-max ranges. Episode
logs remain in `training.monitor.csv`. These curves measure different policies.
A few saved successful games are not a substitute for all-seed metrics. Run configs
also record dependency versions and the environment source hash; evaluating a run
after changing the environment implementation is rejected.

### 4. Watch a frozen learned policy

```bash
uv run python optics_experiment.py render \
  --run experiments/runs-single/starts-16/seed-0 \
  --split test --index 0 --output artifacts/ppo.html
```

Exports portable HTML playback, the final SVG, and every frame in a JSON file. It
uses the exact saved puzzle and selected checkpoint, not a retrained model. Scrub
through steps to see mirror translations, rotations, and the beam after each action.
Keep test playback for final reporting rather than tuning the policy to test poses.

## Custom puzzles and future ablations

A puzzle JSON has `name`, `source`, `source_angle`, `target`, `target_radius`,
`mirror_length`, `solution` (one `(x,y,theta)` per mirror) and `obstacles` rectangles.
Pass that path to `prepare --puzzle path.json` or `optics_play.py --puzzle path.json`.
`prepare` verifies the solution illuminates the target before generating starts.
Save distinct output directories for different puzzles, shaping scales or horizons.
These are separate ablations: changing them along with start diversity would
confound the main question. Models have fixed observation/action dimensions; they
cannot interchange arbitrary numbers of mirrors or obstacles.

## Verification

```bash
uv run python -m unittest discover -s tests -v
```

The suite covers Gymnasium compliance, seeded reproducibility, known optical
solutions, obstacle occlusion, terminal versus truncated rewards, illegal geometry,
pose-bank separation, and HTML serialization. A short PPO training → checkpoint
selection → test table → plot → playback run was also checked. It is a plumbing
check, not evidence that the full proposed experiment has learned a good solver.

API references: [Gymnasium custom environments](https://gymnasium.farama.org/introduction/create_custom_env/)
and [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).
