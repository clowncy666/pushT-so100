# SO-100 PushT — Diffusion Policy with C++ Middleware Inference

A complete imitation-learning pipeline for the **PushT** manipulation task on the **SO-100 robotic arm** in MuJoCo. The key engineering contribution is a **cross-language inference architecture**: Python gymnasium environment ↔ pybind11 ↔ [TinyMiddleware](https://github.com/clowncy/TinyMiddleware) C++ EventLoop/DDS ↔ TensorRT U-Net runner.

[中文版](README-ZH.md)

![Demo](assets/image.png)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│  Python Side (this repo)                                 │
│                                                          │
│  env_gym_ee.py ──► infer_trt.py                         │
│       │                │                                 │
│       │         tinymiddleware_py.so (pybind11)          │
│       │                │                                 │
└───────┼────────────────┼─────────────────────────────────┘
        │                │ VisionMsg (Fast DDS SHM)
        │         ┌──────▼──────────────────────┐
        │         │  TinyMiddleware C++ Node     │
        │         │  EventLoop (epoll reactor)   │
        │         │  ThreadPool → TRT U-Net      │
        │         │  ActionMsg → callback        │
        │         └──────────────────────────────┘
        │
  MuJoCo env.step(action)
```

The Python side publishes camera frames as `VisionMsg` via Fast DDS shared memory; the C++ node receives them, runs TensorRT inference, and sends back `ActionMsg`. The Python callback receives the action and steps the simulator — decoupling simulation from inference scheduling.

---

## Project Structure

```
├── chernyadev .../trs_so_arm100/   # MuJoCo MJCF model (SO-100 arm + T-block scene)
├── script/
│   ├── record_demonstration_data.sh
│   ├── train_policy.sh
│   └── infer.sh
├── src/
│   ├── env_human_ee.py       # Joystick teleoperation for data collection
│   ├── env_gym_ee.py         # Gymnasium env (absolute EE position action)
│   ├── env_gym_ee_stable.py  # Gymnasium env (normalized [-1,1] action + wider tolerances)
│   ├── helper.py             # Pose matching utilities
│   ├── train.py              # DiffusionPolicy training (LeRobot 4.x)
│   ├── infer.py              # Inference via TinyMiddleware callback (no normalization)
│   ├── infer_stable.py       # Inference + dynamic stats extraction & denormalization
│   ├── infer_trt.py          # Same as infer_stable but with TRT engine (recommended)
│   ├── export_all.py         # Export DiffusionPolicy → ONNX (vision encoder + U-Net)
│   ├── build_engine.py       # Compile ONNX → TensorRT FP16 engine
│   └── onnx_models/          # Generated ONNX / TRT files (gitignored, large binaries)
├── assets/
├── environment.yml
└── README.md
```

---

## Workflow

### 1. Environment Setup

```bash
conda env create -f environment.yml
conda activate pusht
```

Requires: `mujoco`, `gymnasium`, `lerobot>=4.0`, `pybind11`, `tensorrt` (for TRT path).

### 2. Data Collection

Use a gamepad to teleoperate the arm in MuJoCo and record demonstrations:

```bash
./script/record_demonstration_data.sh
# or directly:
cd src && python env_human_ee.py --repo_id ./data/my_dataset --fps 10
```

Controls: left stick → EE x/y, LB/A → height, right stick → yaw, X → reset, B → start/stop recording.

A pre-collected dataset is available on Hugging Face: [qian1dqs/so100-pusht](https://huggingface.co/datasets/qian1dqs/so100-pusht)

![Teleoperation](assets/image-20260314190824852.png)

### 3. Training

```bash
./script/train_policy.sh
# or:
cd src && python train.py --data-path ./data/my_dataset --training-steps 13000
```

Uses LeRobot's `DiffusionPolicy` (CNN U-Net backbone, ResNet-18 vision encoder). A pretrained checkpoint is at [qian1dqs/so100-pusht-diffusion](https://huggingface.co/qian1dqs/so100-pusht-diffusion).

Loss curve:

![Training loss](assets/image-20260314191742736.png)

### 4. Export to ONNX + TensorRT

```bash
cd src
python export_all.py          # → onnx_models/vision_encoder.onnx + noise_unet.onnx
python build_engine.py        # → onnx_models/vision_encoder.engine + noise_unet.engine
```

`build_engine.py` enables FP16 automatically if the GPU supports it (RTX series). The `.engine` files are GPU-architecture-specific and are gitignored.

### 5. Inference

**Standard (no TRT):**
```bash
cd src && python infer.py
```

**With TRT + normalization stats (recommended):**
```bash
cd src && python infer_trt.py
# or:
./script/infer.sh
```

Both scripts launch a `TinyMiddleware` node in a background thread, publish camera frames via Fast DDS, and wait for action callbacks from the C++ inference process.

Sample inference:

![Demo 1](assets/show1.gif)
![Demo 2](assets/show2.gif)

---

## TinyMiddleware Integration

The `.so` binding (`src/tinymiddleware_py.cpython-310-x86_64-linux-gnu.so`) is compiled from [TinyMiddleware](https://github.com/clowncy/TinyMiddleware)'s `src/python_bindings.cpp` using pybind11.

```python
import tinymiddleware_py

node = tinymiddleware_py.Node("so100_mujoco_bridge")
vision_pub = node.create_vision_publisher("VisionDataTopic")
node.create_action_subscription("ActionDataTopic", action_callback)

threading.Thread(target=node.spin, daemon=True).start()
```

The C++ side handles: Fast DDS SHM transport → epoll EventLoop → ThreadPool → TRT inference → ActionMsg publish. This keeps the Python side purely reactive (callback-driven), avoiding polling overhead.

**Normalization**: `infer_trt.py` extracts `min/max` stats from the pretrained `DiffusionPolicy` config at runtime, maps U-Net output `[-1, 1]` → physical workspace coordinates, and maps joint states into the normalized range the model expects — without hardcoding any constants.

---

## Troubleshooting

- **No `.so` file**: Build TinyMiddleware with `python_bindings.cpp` enabled and copy the output here.
- **TRT engine mismatch**: Re-run `build_engine.py` on the target GPU; engines are not portable across GPU architectures.
- **MuJoCo rendering on headless server**: Set `MUJOCO_GL=egl` before running.
- **LeRobot API changes**: Tested with `lerobot==4.4`; normalization_mapping field names may differ in other versions.

---

## Acknowledgements

- [LeRobot](https://github.com/huggingface/lerobot) — imitation learning framework
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) — policy architecture
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — SO-100 MJCF model
- [TinyMiddleware](https://github.com/clowncy/TinyMiddleware) — C++ middleware (EventLoop / Fast DDS / ThreadPool)
