# SO-100 PushT — 基于 C++ 中间件的 Diffusion Policy 推理架构

> **Fork 自 [boaoqian/pushT-so100](https://github.com/boaoqian/pushT-so100)**
> 原仓库提供训练流程与 MuJoCo 仿真环境。本 Fork 在此基础上新增了 **TensorRT 加速的 C++ 推理后端** 与 **Fast DDS 中间件层**，由 [@clowncy666](https://github.com/clowncy666) 完成。

![Demo](assets/image.png)

---

## 我的贡献

原项目的训练与推理全部在 Python 侧完成。我的工作聚焦于**将推理与仿真解耦**，并构建面向部署的 C++ 推理流水线：

| 工作内容 | 说明 |
|----------|------|
| **跨语言架构** | 通过 pybind11 将 Python gymnasium 环境与 C++ [TinyMiddleware](https://github.com/clowncy666/TinyMiddleware) 节点桥接；定义 `VisionMsg` / `ActionMsg` DDS 消息类型，实现零拷贝共享内存传输 |
| **TensorRT 导出流水线** | 编写 `export_all.py` 将 DiffusionPolicy 的视觉编码器与 U-Net 去噪器导出为 ONNX，再由 `build_engine.py` 编译为 FP16 TRT 引擎；全链路 Python → ONNX → TRT 无需手动干预 |
| **运行时归一化** | 在 `infer_trt.py` 中实现动态参数提取：启动时从预训练模型 config 读取动作的 `min/max` 边界，将 U-Net 输出 `[-1, 1]` 映射到物理工作空间坐标，消除硬编码常量依赖 |
| **事件驱动推理循环** | 将原有轮询式推理替换为回调驱动架构：Python 通过 Fast DDS 共享内存发布 `VisionMsg`；C++ EventLoop（epoll reactor + ThreadPool）执行 TRT 推理后以 `ActionMsg` 回调返回 Python；仿真与推理完全解耦 |

---

## 架构总览

```
┌──────────────────────────────────────────────────────────┐
│  Python 侧（本仓库）                                      │
│                                                          │
│  env_gym_ee.py ──► infer_trt.py                         │
│       │                │                                 │
│       │         tinymiddleware_py.so (pybind11)          │
│       │                │                                 │
└───────┼────────────────┼─────────────────────────────────┘
        │                │ VisionMsg（Fast DDS 共享内存）
        │         ┌──────▼──────────────────────┐
        │         │  TinyMiddleware C++ 节点      │
        │         │  EventLoop（epoll reactor）   │
        │         │  ThreadPool → TRT U-Net      │
        │         │  ActionMsg → 回调             │
        │         └──────────────────────────────┘
        │
  MuJoCo env.step(action)
```

Python 侧将相机帧以 `VisionMsg` 形式通过 Fast DDS 共享内存发布；C++ 节点接收后运行 TensorRT 推理，将 `ActionMsg` 回调返回 Python；Python 回调执行仿真步进——仿真调度与推理调度完全解耦。

---

## 项目结构

```
├── chernyadev .../trs_so_arm100/   # MuJoCo MJCF 模型（SO-100 机械臂 + T 形块场景）
├── script/
│   ├── record_demonstration_data.sh
│   ├── train_policy.sh
│   └── infer.sh
├── src/
│   ├── env_human_ee.py       # 手柄遥操作数据采集
│   ├── env_gym_ee.py         # Gymnasium 环境（绝对末端位置动作空间）
│   ├── env_gym_ee_stable.py  # Gymnasium 环境（归一化 [-1,1] 动作 + 宽容差）
│   ├── helper.py             # 位姿匹配工具
│   ├── train.py              # DiffusionPolicy 训练（LeRobot 4.x）
│   ├── infer.py              # 基于 TinyMiddleware 回调的推理（无归一化）
│   ├── infer_stable.py       # 推理 + 动态 stats 提取与反归一化
│   ├── infer_trt.py          # 同上，使用 TRT 引擎（推荐）
│   ├── export_all.py         # DiffusionPolicy → ONNX 导出
│   ├── build_engine.py       # ONNX → TensorRT FP16 编译
│   └── onnx_models/          # 生成的 ONNX / TRT 文件（已 gitignore，大文件）
├── assets/
├── environment.yml
└── README-ZH.md
```

---

## 工作流程

### 1. 环境配置

```bash
conda env create -f environment.yml
conda activate pusht
```

依赖：`mujoco`、`gymnasium`、`lerobot>=4.0`、`pybind11`、`tensorrt`（TRT 推理路径需要）。

### 2. 数据采集

使用手柄遥操作机械臂在 MuJoCo 中录制演示数据：

```bash
./script/record_demonstration_data.sh
# 或直接运行：
cd src && python env_human_ee.py --repo_id ./data/my_dataset --fps 10
```

手柄映射：左摇杆 → 末端 x/y，LB/A → 高度，右摇杆 → 偏航，X → 重置，B → 开始/停止录制。

Hugging Face 上有预采集数据集可直接使用：[qian1dqs/so100-pusht](https://huggingface.co/datasets/qian1dqs/so100-pusht)

![遥操作界面](assets/image-20260314190824852.png)

### 3. 训练

```bash
./script/train_policy.sh
# 或：
cd src && python train.py --data-path ./data/my_dataset --training-steps 13000
```

使用 LeRobot 的 `DiffusionPolicy`（CNN U-Net 主干，ResNet-18 视觉编码器）。预训练检查点：[qian1dqs/so100-pusht-diffusion](https://huggingface.co/qian1dqs/so100-pusht-diffusion)。

损失曲线：

![训练损失](assets/image-20260314191742736.png)

### 4. 导出 ONNX + TensorRT

```bash
cd src
python export_all.py          # → onnx_models/vision_encoder.onnx + noise_unet.onnx
python build_engine.py        # → onnx_models/vision_encoder.engine + noise_unet.engine
```

`build_engine.py` 在支持 FP16 的 GPU（RTX 系列）上自动启用半精度。`.engine` 文件与 GPU 架构绑定，已加入 `.gitignore`。

### 5. 推理

**标准模式（无 TRT）：**
```bash
cd src && python infer.py
```

**TRT + 归一化参数（推荐）：**
```bash
cd src && python infer_trt.py
# 或：
./script/infer.sh
```

两个脚本均在后台线程启动 TinyMiddleware 节点，通过 Fast DDS 发布相机帧，等待 C++ 推理进程的动作回调。

推理示例：

![Demo 1](assets/show1.gif)
![Demo 2](assets/show2.gif)

---

## TinyMiddleware 集成说明

`.so` 绑定文件（`src/tinymiddleware_py.cpython-310-x86_64-linux-gnu.so`）由 [TinyMiddleware](https://github.com/clowncy666/TinyMiddleware) 的 `src/python_bindings.cpp` 通过 pybind11 编译生成。

```python
import tinymiddleware_py

node = tinymiddleware_py.Node("so100_mujoco_bridge")
vision_pub = node.create_vision_publisher("VisionDataTopic")
node.create_action_subscription("ActionDataTopic", action_callback)

threading.Thread(target=node.spin, daemon=True).start()
```

C++ 侧处理：Fast DDS 共享内存传输 → epoll EventLoop → ThreadPool → TRT 推理 → 发布 ActionMsg。Python 侧保持纯回调驱动，无轮询开销。

**归一化细节**：`infer_trt.py` 在启动时从预训练 `DiffusionPolicy` config 中提取动作的 `min/max` 统计值，将 U-Net 输出 `[-1, 1]` 映射回物理工作空间坐标，同时将关节状态归一化到模型期望的输入范围——全程无硬编码常量。

---

## 常见问题

- **找不到 `.so` 文件**：在目标机器上编译 TinyMiddleware（启用 `python_bindings.cpp`）并将产物复制到此处。
- **TRT 引擎不兼容**：在目标 GPU 上重新运行 `build_engine.py`；引擎文件不可跨 GPU 架构移植。
- **无头服务器 MuJoCo 渲染问题**：运行前设置 `MUJOCO_GL=egl`。
- **LeRobot API 变动**：在 `lerobot==4.4` 下测试通过；其他版本的归一化字段名称可能有差异。

---

## 致谢

- [LeRobot](https://github.com/huggingface/lerobot) — 模仿学习框架
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) — 策略架构
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — SO-100 MJCF 模型
- [TinyMiddleware](https://github.com/clowncy666/TinyMiddleware) — C++ 中间件（EventLoop / Fast DDS / ThreadPool）
