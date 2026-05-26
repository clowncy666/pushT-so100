import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import logging
import queue
import time
import threading
import cv2
import numpy as np

# 💥 官方原生环境
from env_gym_ee import PushT
from gymnasium.wrappers import RecordVideo

import tinymiddleware_py
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

action_queue = queue.Queue(maxsize=1)

def action_callback(msg):
    norm_x = float(np.clip(msg.target_x, -1.0, 1.0))
    norm_y = float(np.clip(msg.target_y, -1.0, 1.0))
    action = np.array([norm_x, norm_y], dtype=np.float32)
    if action_queue.full():
        try: action_queue.get_nowait()
        except queue.Empty: pass
    action_queue.put(action)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_path", type=str, default="chernyadev mujoco_menagerie add-so-arm100 trs_so_arm100/human_env.xml")
    parser.add_argument("--video_folder", type=str, default="outputs/recorded_videos")
    args = parser.parse_args()

    # ================= 1. 安全读取真正的模型极值 =================
    logger.info("正在加载真实模型参数...")
    policy = DiffusionPolicy.from_pretrained("qian1dqs/so100-pusht", device="cpu")
    
    # State 参数
    has_state_stats = False
    if "observation.state" in policy.config.normalization_mapping:
        state_stats = policy.config.normalization_mapping["observation.state"]
        s_min = np.array(state_stats["min"], dtype=np.float32)
        s_max = np.array(state_stats["max"], dtype=np.float32)
        has_state_stats = True
    
    # Action 参数 (💥 核心中的核心！再也不瞎猜了)
    has_action_stats = False
    if "action" in policy.config.normalization_mapping:
        action_stats = policy.config.normalization_mapping["action"]
        a_min = np.array(action_stats["min"], dtype=np.float32)
        a_max = np.array(action_stats["max"], dtype=np.float32)
        has_action_stats = True
        logger.info(f"✅ 成功提取真实 Action 物理边界: min={a_min}, max={a_max}")

    # ================= 2. 启动中间件 =================
    mw_node = tinymiddleware_py.Node("so100_mujoco_bridge")
    vision_pub = mw_node.create_vision_publisher("VisionDataTopic")
    mw_node.create_action_subscription("ActionDataTopic", action_callback)
    threading.Thread(target=mw_node.spin, daemon=True).start()
    
    # ================= 3. 启动官方环境 =================
    raw_env = PushT(xml_path=args.env_path, render_mode="rgb_array")
    env = RecordVideo(raw_env, video_folder=args.video_folder, name_prefix="trt_pusht")
    obs, _ = env.reset()

    logger.info("🚀 C++ 联合推理开始...")
    frame_counter = 0 

    try:
        while True:
            frame_rgb = cv2.resize(obs["cam_top"], (224, 224), interpolation=cv2.INTER_AREA)
            raw_state = np.array([
                obs["Rotation"], obs["Pitch"], obs["Elbow"],
                obs["Wrist_Pitch"], obs["Wrist_Roll"]
            ], dtype=np.float32)
            
            if has_state_stats:
                norm_state = 2.0 * (raw_state - s_min) / (s_max - s_min) - 1.0
                norm_state = np.clip(norm_state, -1.0, 1.0).astype(np.float32)
            else:
                norm_state = raw_state
            
            frame_counter += 1
            msg = tinymiddleware_py.VisionMsg()
            msg.frame_id = frame_counter
            msg.width, msg.height, msg.channels = 224, 224, 3
            msg.payload = frame_rgb.tobytes() + norm_state.tobytes()
            vision_pub.publish(msg)
            
            try:
                raw_unet_action = action_queue.get(timeout=0.5)
                # 💥 最终完美映射！将网络的 [-1, 1] 精准转换为物理空间坐标
                if has_action_stats:
                    action_to_exec = (raw_unet_action + 1.0) / 2.0 * (a_max - a_min) + a_min
                else:
                    action_to_exec = raw_unet_action
            except queue.Empty:
                action_to_exec = np.zeros(2, dtype=np.float32)
            
            obs, reward, terminated, truncated, info = env.step(action_to_exec)
            
            if frame_counter % 20 == 0:
                logger.info(f"帧 {frame_counter} | 网络: {raw_unet_action} | 物理坐标: {action_to_exec}")

            if terminated or truncated:
                logger.info("🎯 任务成功完成！" if terminated else "⌛ 任务超时。")
                break
                
            cv2.imshow("PushT C++ Brain", cv2.cvtColor(obs["cam_top"], cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"): break

    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()