import os
# 💥 强行使用国内镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import logging
import queue
import time
import threading
import cv2
import numpy as np

from env_gym_ee_stable import PushTStable
from gymnasium.wrappers import RecordVideo

# ============================================================================
# [基建引入] 导入自研的 C++ 中间件 Python 绑定
# ============================================================================
import tinymiddleware_py
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# 使用阻塞队列代替 Lock 和列表
action_queue = queue.Queue(maxsize=1)
callback_counter = 0

def action_callback(msg):
    """
    C++ 端的线程池收到 ActionMsg 后会通过这个函数通知 Python
    """
    global callback_counter
    callback_counter += 1
    
    # 接收 C++ 发来的原始动作数值 (U-Net 输出的归一化动作，一般在 [-1, 1])
    norm_x = float(np.clip(msg.target_x, -1.0, 1.0))
    norm_y = float(np.clip(msg.target_y, -1.0, 1.0))
    
    action = np.array([norm_x, norm_y], dtype=np.float32)

    # 填入队列 (丢弃旧动作，保持最新)
    if action_queue.full():
        try:
            action_queue.get_nowait()
        except queue.Empty:
            pass
    action_queue.put(action)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_path", type=str, default="chernyadev mujoco_menagerie add-so-arm100 trs_so_arm100/human_env.xml")
    parser.add_argument("--video_folder", type=str, default="outputs/recorded_videos")
    return parser.parse_args()

def main():
    args = parse_args()

    # ========================================================================
    # [模型 Stats 提取：手撕归一化与反归一化]
    # ========================================================================
    logger.info("正在加载 LeRobot 策略以提取归一化参数 (Stats)...")
    policy = DiffusionPolicy.from_pretrained("qian1dqs/so100-pusht", device="cpu")
    policy.eval()
    
    # 1. 获取 State 的参数
    has_state_stats = False
    if "observation.state" in policy.config.normalization_mapping:
        state_stats = policy.config.normalization_mapping["observation.state"]
        s_min = np.array(state_stats["min"], dtype=np.float32)
        s_max = np.array(state_stats["max"], dtype=np.float32)
        has_state_stats = True
        logger.info("✅ 成功提取 State 归一化参数。")
    
    # 2. 获取 Action 的参数 (新增)
    has_action_stats = False
    if "action" in policy.config.normalization_mapping:
        action_stats = policy.config.normalization_mapping["action"]
        a_min = np.array(action_stats["min"], dtype=np.float32)
        a_max = np.array(action_stats["max"], dtype=np.float32)
        has_action_stats = True
        logger.info("✅ 成功提取 Action 反归一化参数。")

    # ========================================================================
    # [中间件初始化]
    # ========================================================================
    logger.info("正在启动 TinyMiddleware 异步控制节点...")
    mw_node = tinymiddleware_py.Node("so100_mujoco_bridge")
    
    vision_pub = mw_node.create_vision_publisher("VisionDataTopic")
    mw_node.create_action_subscription("ActionDataTopic", action_callback)
    
    spin_thread = threading.Thread(target=mw_node.spin, daemon=True)
    spin_thread.start()
    
    frame_counter = 0 
    
    raw_env = PushTStable(xml_path=args.env_path, render_mode="rgb_array")
    env = RecordVideo(raw_env, video_folder=args.video_folder, name_prefix="callback_pusht_stable")
    obs, _ = env.reset()

    logger.info("🚀 联调开始：大脑已上线，开始实时推理循环...")

    try:
        while True:
            # 1. 获取 RGB 画面并 Resize
            frame_rgb = obs["cam_top"]
            frame_rgb = cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_AREA)
            
            # 2. 提取原始关节状态
            raw_state = np.array([
                obs["Rotation"], obs["Pitch"], obs["Elbow"],
                obs["Wrist_Pitch"], obs["Wrist_Roll"]
            ], dtype=np.float32)
            
            # 3. 💥 手撕正向归一化 (输入给网络)
            if has_state_stats:
                norm_state = 2.0 * (raw_state - s_min) / (s_max - s_min) - 1.0
                norm_state = np.clip(norm_state, -1.0, 1.0).astype(np.float32)
            else:
                norm_state = raw_state
            
            frame_counter += 1
            msg = tinymiddleware_py.VisionMsg()
            msg.frame_id = frame_counter
            msg.width, msg.height, msg.channels = 224, 224, 3
            
            # 打包图像和状态发送给 C++
            msg.payload = frame_rgb.tobytes() + norm_state.tobytes()
            vision_pub.publish(msg)
            
            start_wait = time.time()
            
            # 4. 等待 C++ 返回动作
            try:
                raw_unet_action = action_queue.get(timeout=0.5)
                
                # 💥 手撕反向归一化 (还原为物理动作)
                if has_action_stats:
                    # 公式: x = (y + 1) / 2 * (max - min) + min
                    action_to_exec = (raw_unet_action + 1.0) / 2.0 * (a_max - a_min) + a_min
                else:
                    action_to_exec = raw_unet_action
                    
            except queue.Empty:
                logger.warning(f"帧 {frame_counter} 推理超时...")
                action_to_exec = np.zeros(2, dtype=np.float32)
            
            # 5. 执行动作
            obs, reward, terminated, truncated, info = env.step(action_to_exec)
            
            if frame_counter % 20 == 0:
                delay = (time.time() - start_wait) * 1000
                logger.info(f"帧 {frame_counter} | 延迟: {delay:.1f}ms | 网络输出: {raw_unet_action} | 实际下发: {action_to_exec}")

            if terminated or truncated:
                logger.info(f"任务完成或超时: terminated={terminated}, truncated={truncated}")
                break
                
            frame_bgr = cv2.cvtColor(obs["cam_top"], cv2.COLOR_RGB2BGR)
            cv2.imshow("PushT Callback Monitor", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        logger.info("用户停止。")
    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()