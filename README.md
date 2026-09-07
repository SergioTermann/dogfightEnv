# dogfightEnv

**Air-combat simulation for reinforcement learning and mission-level decision making.**

dogfightEnv combines JSBSim six-degree-of-freedom flight dynamics, Harfang3D visualization, Gym-style environments, and a rule-based or LLM-powered mission commander. Train flight policies, evaluate tactical assignments, and inspect engagements in a shared 3D simulation.

![dogfightEnv air-combat simulation](docs/images/readme/cover.jpg)

[Quick Start](#quick-start) | [RL Training](#rl-training) | [Mission Commander](#mission-commander) | [Network Protocol](dogfight_sandbox_hg2/documentation_network.md) | [Issues](https://github.com/SergioTermann/dogfightEnv/issues)

## Features

| Area | What is included |
| --- | --- |
| Flight dynamics | JSBSim 1.2.1, F-16 aerodynamics, afterburner, stability augmentation, and a fixed 1/60 s flight-dynamics step |
| Visualization | Harfang3D ocean, terrain, clouds, HUD, multiple cameras, and flight-path prediction |
| RL environments | Legacy Gym-style interfaces; the reference 1v1 environment exposes 25 observations and 5 continuous actions |
| Training | Native PyTorch PPO, SAC, and Rainbow, with observation normalization, checkpoints, and JSONL metrics |
| Mission command | Rule-based or OpenAI-compatible LLM decisions: `engage`, `patrol`, `retreat`, and `hold` |
| Manual flight | Keyboard and Xbox gamepad controls, with environments for collecting expert demonstrations |
| Experiment inspection | In-engine views, commander decision logs, and Tacview ACMI output from the 1v1 environment |

## Architecture

The sandbox owns the simulation state, aircraft, missiles, and rendering. An external Python client exchanges state and commands with it over JSON over TCP.

![Architecture showing the sandbox, local controls, and external clients](docs/images/readme/architecture.svg)

**One sandbox instance accepts one TCP client at a time.** Choose either an RL session or a Commander session. Keyboard and gamepad input run locally inside the sandbox and do not occupy the network connection.

![Execution flow for RL training and mission command](docs/images/readme/execution.svg)

RL clients advance the scene through `update_scene`. The Commander polls state and sends changed assignments while the sandbox runs freely; it does not control the simulation clock. Restart the sandbox when switching between these workflows.

## Quick Start

### Prerequisites

| Component | Requirement |
| --- | --- |
| Sandbox platform | Windows 10 or 11, 64-bit |
| Sandbox runtime | Bundled embedded Python 3.8 and native libraries under `dogfight_sandbox_hg2/bin/` |
| RL client | A separate Python environment with Gym, NumPy, PyTorch, Harfang, and PrettyTable |
| Commander | Bundled Python or a compatible system Python; no third-party packages required |
| Graphics and assets | A graphics-capable machine and the sandbox asset directories described below |
| Training acceleration | CUDA is optional; CPU training is supported |

**A Git clone does not include the full graphics assets.** Before launching, provision both `dogfight_sandbox_hg2/source/assets/` and `dogfight_sandbox_hg2/source/assets_compiled/`. These directories are excluded from Git. The [upstream sandbox releases](https://github.com/harfang3d/dogfight-sandbox-hg2/releases) and [upstream setup notes](dogfight_sandbox_hg2/README.md#how-to-run-dogfight) describe the original distribution; ensure any assets you supply match this checkout's scene references and runtime.

Commands below use PowerShell. Start each workflow from the repository root unless a command explicitly changes directories.

### 1. Start the sandbox

In a dedicated terminal:

```powershell
cd dogfight_sandbox_hg2/source
..\bin\python\python.exe main.py auto_network mission=1
```

| Argument | Network mission |
| --- | --- |
| `mission=1` | 1v1 |
| `mission=2` | 2v2 |
| `mission=3` | 3v3 |

Read the **HOST / PORT** shown in the upper-left corner of the sandbox window. The default port is `50888`. The server binds to the address resolved from the machine's hostname, so use the displayed address even when the client runs on the same machine.

### 2. Prepare the RL client

In a second terminal at the repository root, create or activate your Python environment, then install:

```powershell
python -m pip install -r requirements-train.txt
python -m pip install harfang prettytable
```

`requirements-train.txt` lists PyTorch, NumPy, and Gym. The root environment modules also import Harfang and PrettyTable, so these are required for the RL client even when rendering is disabled. Choose a Python version supported by all of these packages; the embedded sandbox interpreter is separate from the training environment.

For a Commander-only session, skip the RL dependencies and continue to [Mission Commander](#mission-commander).

### 3. Connect to the 1v1 environment

Run this Python example from the repository root. Replace `<SANDBOX_HOST>` with the address shown in the sandbox:

```python
from oneVSoneEnv import oneVSoneEnv
from training.wrapper import EnvAdapter

env = EnvAdapter(
    oneVSoneEnv(host="<SANDBOX_HOST>", port="50888", rendering=True),
    normalize=False,
)
obs = env.reset()

for _ in range(100):
    action = env.action_space.sample()  # [roll, pitch, yaw, thrust, fire]
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
```

The environments use the **legacy Gym API**: `reset()` returns an observation and `step()` returns `(obs, reward, done, info)`. They do not implement the Gymnasium reset/step contract. `EnvAdapter` converts observations to NumPy arrays and performs a neutral step after reset to refresh the initial state.

## RL Training

Start the sandbox first. In your client terminal, set the displayed host and run **one** training command:

```powershell
$sandboxHost = "<SANDBOX_HOST>"

python -m training.train --algo ppo --env oneVSone --host $sandboxHost --timesteps 500000
python -m training.train --algo sac --env oneVSone --host $sandboxHost --timesteps 1000000
python -m training.train --algo rainbow --env oneVSone --host $sandboxHost --timesteps 1000000
```

| Algorithm | Action representation | Implementation |
| --- | --- | --- |
| PPO | Continuous | Gaussian policy, generalized advantage estimation, clipped objective |
| SAC | Continuous | Twin Q networks, automatic entropy temperature, soft target updates |
| Rainbow | Discrete grid over continuous controls | n-step returns, Double Q, dueling networks, NoisyNet, prioritized replay, C51 |

The training CLI registers these environment names:

| CLI name | Implementation |
| --- | --- |
| `oneVSone` | [oneVSoneEnv.py](oneVSoneEnv.py) |
| `twoVSone` | [twoVStwo.py](twoVStwo.py) |
| `ia_enemy` | [IA_enemy_env.py](IA_enemy_env.py) |

The `twoVSone` CLI name is intentional in the current registry, despite its implementation filename. Use `oneVSone` with `mission=1` for the introductory workflow; other wrappers have their own aircraft assumptions.

### Options and checkpoints

| Option | Purpose |
| --- | --- |
| `--host`, `--port` | Select the sandbox address; always set the host for your machine |
| `--render` | Request the 3D view during training; omitted by default |
| `--device cpu` or `--device cuda` | Select the training device; automatic selection is the default |
| `--seed 1` | Set the NumPy and PyTorch seed |
| `--set lr=1e-4 gamma=0.995` | Override algorithm hyperparameters |
| `--name experiment_01` | Set a run name and output subdirectory |
| `--no-normalize` | Disable observation normalization |

Outputs are written to `checkpoints/<algo>_<env>/` by default:

```text
checkpoints/ppo_oneVSone/
    model_final.pt       # Final policy checkpoint
    extra_final.pt       # Environment, normalization, and action-grid metadata
    model_best.pt        # Best recent training return, when recorded
    extra_best.pt        # Metadata paired with the best checkpoint
    model_<step>.pt       # Periodic checkpoint
    extra_<step>.pt       # Matching metadata
    log.jsonl            # Episode returns, lengths, throughput, and update metrics
```

Keep each `model_*.pt` with its matching `extra_*.pt`. The best checkpoint is selected from recent training returns at logging intervals; it is not a separate evaluation score and may be absent in short runs.

To watch a completed run in a fresh sandbox session:

```powershell
python -m training.enjoy --model checkpoints/ppo_oneVSone/model_final.pt --host $sandboxHost --episodes 3
```

## Mission Commander

Use a fresh sandbox session, for example with `mission=2` for 2v2. Set `host` and `port` in [llm_commander/config.json](llm_commander/config.json) to the displayed server address, then run from the repository root:

```powershell
.\dogfight_sandbox_hg2\bin\python\python.exe llm_commander/commander.py
```

The default `rule` engine requires no API key. It assigns `engage`, `patrol`, `retreat`, or `hold` tasks to aircraft on the configured side.

| Configuration | Meaning |
| --- | --- |
| `engine` | `rule` or `llm` |
| `side` | `allies` or `ennemies`; preserve the spelling used by the protocol |
| `decision_period_s` | Decision interval; default `10` seconds |
| `poll_interval_s` | State polling interval; default `0.5` seconds |
| `blue_ia` | Enable the built-in AI for the opposing blue side when commanding `ennemies` |

The CLI also accepts `--once` for a single decision cycle, `--duration 60` for a timed session, and `--config path/to/config.json` for an alternate configuration. `--dry-run` skips applying the commander's task assignments, but still connects to the sandbox; configured opposing-side AI activation can still occur during connection.

### Use an LLM

In the Commander configuration:

1. Set `engine` to `llm`.
2. Set `llm.api_base` to the **full OpenAI-compatible chat completions URL**, including the endpoint path.
3. Set `llm.api_key` and `llm.model` for your provider.

An empty API key falls back to the rule engine. Network, API, or parsing failures retain the previous plan. Decision records are written to `llm_commander/decisions.jsonl`.

## Physics and Visualization

Configure the sandbox in [dogfight_sandbox_hg2/config.json](dogfight_sandbox_hg2/config.json):

| Setting | Default | Purpose |
| --- | --- | --- |
| `Physics.engine` | `jsbsim` | Select `jsbsim` or the original `legacy` physics |
| `FlightPrediction.enabled` | `true` | Display predicted flight paths |
| `FlightPrediction.horizon_s` | `10` | Prediction horizon in seconds |
| `FlightPrediction.steps` | `20` | Prediction sampling steps |

All current aircraft types share the JSBSim F-16 aerodynamic data. Their different visual models do not imply distinct flight dynamics. See [jsbsim_flight_model.py](dogfight_sandbox_hg2/source/jsbsim_flight_model.py) for model mappings and control conventions. Missiles retain the sandbox's proportional-navigation model.

![In-engine views of mission assignments, flight prediction, external flight, and the cockpit](docs/images/readme/simulation.jpg)

### Controls

| Control | Keys |
| --- | --- |
| Pitch / roll | Arrow keys |
| Increase / decrease throttle | `Home` / `End` |
| Toggle afterburner | `Space` |
| Fire gun / missile | `Enter` / `F1` |
| Select next target / toggle landing gear | `T` / `G` |
| Increase / decrease airbrake | `B` / `N` |
| Increase / decrease flaps | `C` / `V` |
| Built-in AI / autopilot / easy steering | `I` / `A` / `E` |
| Rear / front / left / right view | Numpad `2` / `8` / `4` / `6` |
| Satellite / cockpit / next tracked aircraft | Numpad `5` / `3` / `1` |
| Decrease / increase field of view | `Insert` / `PageUp` |

See the [English gamepad mapping](docs/images/gamepad_mapping_en.svg) for controller bindings.

## Repository Layout

```text
dogfightEnv/
    dogfight_sandbox_hg2/
        bin/                      # Embedded Windows runtime and native libraries
        source/                   # Simulation, physics, rendering, and network server
        network_client_example/   # Python TCP client and examples
        tools/                    # Sandbox integration checks
        config.json               # Graphics, physics, and prediction settings
    training/                     # Algorithms, wrappers, training, and playback
    llm_commander/                 # Commander, tacticians, and configuration
    docs/                         # Diagrams, screenshots, and artwork tools
    oneVSoneEnv.py                 # Reference 1v1 environment
    twoVStwo.py                    # Wrapper registered as twoVSone
    IA_enemy_env.py                # AI-opponent environment
    human_expert_env.py            # Expert-demonstration environment
    requirements-train.txt         # Core training dependencies
```

## Validation

Run the algorithm tests from the repository root with your training Python environment. These tests do not require a sandbox:

```powershell
python -m training.tests.test_algos_toy
```

The integration checks require the sandbox runtime and assets. Run them separately:

| Check | Command from the repository root | Setup |
| --- | --- | --- |
| Flight dynamics | `.\dogfight_sandbox_hg2\bin\python\python.exe dogfight_sandbox_hg2/tools/test_jsbsim_physics.py` | Start a 1v1 sandbox first; set `$env:DOGFIGHT_HOST` and optionally `$env:DOGFIGHT_PORT` |
| Commander | `.\dogfight_sandbox_hg2\bin\python\python.exe dogfight_sandbox_hg2/tools/test_llm_commander.py` | Starts its own 2v2 sandbox; uses the host and port in the Commander config |
| Training smoke test | `python dogfight_sandbox_hg2/tools/test_training_smoke.py --host $sandboxHost` | Starts its own 1v1 sandbox; requires the RL client dependencies |

Close existing sandbox instances before running checks that launch their own server. The Commander test replaces `decisions.jsonl`; the training smoke test resets `checkpoints/_smoke/`.

## Troubleshooting and Limitations

| Symptom or constraint | What to check |
| --- | --- |
| Client waits indefinitely for a connection | Start a network mission and use the displayed host and port. Check firewall access and whether another client occupies the connection. |
| Sandbox fails to load scenes or textures | Confirm both asset directories are populated and launch `main.py` from `dogfight_sandbox_hg2/source/`. |
| RL import fails for `harfang` or `prettytable` | Install the additional client packages in the same Python environment used for training. |
| Code expects Gymnasium return values | Use the documented legacy Gym contract or provide an explicit adapter. |
| No `model_best.pt` after a short run | Use `model_final.pt`; best-model saves depend on completed episodes and logging intervals. |
| Commander and training cannot connect together | Run separate sessions or separate sandbox instances with distinct ports. |

The 1v1 environment writes Tacview-formatted data to `trained_epoch_0.txt` in the working directory and resets the file on episode reset. Preserve an episode's output before starting another if you need it for analysis.

Current development priorities include distributing complete assets, supporting concurrent external clients, adding aircraft-specific JSBSim models, and defining standardized evaluation scenarios. These are planned improvements, not current capabilities.

## License and Acknowledgements

The sandbox in `dogfight_sandbox_hg2/` is derived from [harfang3d/dogfight-sandbox-hg2](https://github.com/harfang3d/dogfight-sandbox-hg2) and is covered by its [GPL-3.0 license](dogfight_sandbox_hg2/LICENSE).

Thanks to Harfang Technologies and [mrwangyou/DBRL](https://github.com/mrwangyou/DBRL). This project builds on [Harfang3D](https://harfang3d.com/), [JSBSim](https://github.com/JSBSim-Team/jsbsim), and [PyTorch](https://pytorch.org/).
