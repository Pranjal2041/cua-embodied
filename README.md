# Computer-Use Agents Are Embodied Generalists

Code for the paper by Pranjal Aggarwal and Sean Welleck.
[Paper](https://pranjal2041.github.io/cua-embodied/paper.pdf) · [Website](https://pranjal2041.github.io/cua-embodied/)

We keep one computer-use agent fixed and build software for each embodied task.
The agent sees screenshots and acts with a mouse and a keyboard. Each task's
software renders the agent's view of the environment and provides controls for
acting in it. With only the software changed, the same agent entered four ECCV
2026 workshop competitions: VLNVerse, DialNav, OpenSUN3D and MeViSv2-Text.

## How it works

- `cua_embodied/agent.py`: the agent. One system prompt and one Gemini model.
  It calls exactly one tool per turn. It keeps the full interaction text. When
  the context holds twenty images, the oldest ten are replaced by a placeholder.
  Videos always stay in context.
- `cua_embodied/computer/`: the agent's only two tools, `mouse` and `keyboard`.
  A call can carry up to 16 inputs. Each input waits for the visible operation
  to finish, and one screenshot is returned after the batch. Mouse coordinates
  are normalized to 0 to 1000 on each axis.
- `cua_embodied/apps/`: one package per competition. Each app has a backend
  that implements the task operations, the task instructions, and the hooks the
  GUI needs. Apps are imported on demand, so you only need the dependencies of
  the apps you use.
- `cua_embodied/gui/`: a local web server and page that turn the backend
  operations into buttons, shortcuts and clickable views. The agent never calls
  its HTTP endpoints. It only sees the page.
- `cua_embodied/run.py`: runs the agent on one task and saves the trace, every
  screenshot and the prediction.

The agent receives the task description, the task instructions and any
reference videos. After every action it receives a screenshot and any video an
operation produced, such as a tracked-mask overlay. It finishes by pressing the
save or stop control in the GUI, which makes the software export the prediction.

### The four apps

| App | Software | Backend models and services |
|---|---|---|
| `vlnverse` | First-person view of a 3D home. Move forward, back, sideways and up or down, turn and tilt, set step sizes, stop and save. | Isaac Sim renderer (`apps/vlnverse/isaac_bridge.py`) |
| `dialnav` | 360-degree panorama with numbered exits, a target clue and a message panel to the Guide. | The Guide, a separate Gemini context with a scene atlas and route previews (`apps/dialnav/guide.py`) |
| `opensun3d` | Colored Faro point cloud with a free camera, jumps to recorded video poses, point selection, four-view candidate review, refinement and saving. | PinPoint3D (`apps/opensun3d/pinpoint_server.py`) and a VOLT model fine-tuned on SceneFun3D (`apps/opensun3d/volt/`) |
| `mevis` | Video player, frame selector, adding, refining and removing objects, full-video tracking and review, saving or no-target. | SAM3 (`apps/mevis/sam3_interface.py`) |

### GUI or text API

`--interface api` runs the comparison arm from the paper. The agent then calls
the same backend operations directly as typed tools (`apps/*/api.py`) and
receives structured feedback. Both arms use the same model, task inputs,
backend and starting view.

## Setup

Python 3.10 or newer, Node.js 18 or newer and Google Chrome.

```bash
pip install -e .            # core: agent, computer, GUI
pip install -e ".[mevis]"   # plus the libraries for an app, or ".[all]"
npm install                 # Puppeteer for the mouse and keyboard driver
cp configs/tasks.example.json configs/tasks.json
echo "GEMINI_API_KEY=..." > .env
```

Then fill in `configs/tasks.json` for the apps you use.

**VLNVerse.** Install Isaac Sim 4.5 and download the VLNVerse scenes
(`Eyz/VLNVerse_scene`) and test splits (`Eyz/VLNVerse_data`). Build the episode
manifest with `python -m cua_embodied.apps.vlnverse.build_manifest`. Start the
renderer with Isaac Sim's own Python before opening a task:

```bash
<isaac-sim>/python.sh cua_embodied/apps/vlnverse/isaac_bridge.py \
    --scene <scene>/start_result_navigation.usd --port 8793
```

**DialNav.** Download the RAIN episodes, the Matterport3D connectivity graphs
and the Matterport3D panoramas. Set `split`, `connectivity_dir` and `rgb_dir`.
The Guide uses `guide_model` and calls the Gemini API on every message.

**OpenSUN3D.** Download the SceneFun3D scene (laser scan, crop mask and
`hires_poses.traj` for each video). `reference_video_dir` holds one
`<video_id>.mp4` per video of the scene, converted from the ARKitScenes
`<video_id>.mov`. Set up PinPoint3D in its own environment with
MinkowskiEngine and point `pinpoint.python`, `pinpoint.repo` and
`pinpoint.checkpoint` to it. On Windows, run it inside WSL by setting
`pinpoint.wsl_distro`. For VOLT proposals, check out VOLT at the commit in
`apps/opensun3d/volt/model.py` and build the per-scene prediction cache with
`python -m cua_embodied.apps.opensun3d.volt.build_cache`.

**MeViSv2.** Download the MeViSv2 validation data and the SAM3 checkpoint, and
install SAM3.

## Usage

Use a task GUI yourself:

```bash
python -m cua_embodied.gui.server --config configs/tasks.json
# open http://127.0.0.1:8808/?app=mevis
```

Run the agent on a task through the GUI:

```bash
python -m cua_embodied.run --app opensun3d --output outputs/runs/opensun3d \
    --model gemini-3.6-flash
```

Add `--interface api` for the text-API arm. Each run writes `task.json`,
`trace.json`, `result.json`, the raw model responses, every screenshot,
`prediction.json` and `summary.json` to the output directory. The backend's own
files, such as masks, trajectories and Guide records, are under `sessions/` or
`backend/`.

## Tests

These run without model calls or GPUs.

```bash
python -m unittest discover -s tests
node --test tests/test_browser_inputs.cjs
```
