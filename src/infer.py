import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
import time
import threading

from env_gym_ee import PushT
from gymnasium.wrappers import RecordVideo

# ============================================================================
# [基建引入] 导入自研的 C++ 中间件 Python 绑定
# ============================================================================
import tinymiddleware_py

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# 全局变量：用于在 C++ 回调和 Python 主循环之间传递动作
# 使用列表包装以便在回调中修改
latest_action = [None] 
action_lock = threading.Lock()

def action_callback(msg):
    """
    C++ 端的线程池收到 ActionMsg 后会通过这个函数通知 Python
    """
    global latest_action
    with action_lock:
        # 将 C++ 传回的反归一化坐标存入缓冲区
        latest_action[0] = np.array([msg.target_x, msg.target_y], dtype=np.float32)

def parse_args():
    parser = argparse.ArgumentParser(description="Run C++ Callback Powered Inference")
    parser.add_argument("--env_path", type=str, default="chernyadev mujoco_menagerie add-so-arm100 trs_so_arm100/human_env.xml")
    parser.add_argument("--video_folder", type=str, default="outputs/recorded_videos")
    return parser.parse_args()

def main():
    args = parse_args()

    # ========================================================================
    # [中间件初始化]
    # ========================================================================
    logger.info("正在启动 TinyMiddleware 异步控制节点...")
    mw_node = tinymiddleware_py.Node("so100_mujoco_bridge")
    
    # 1. 创建视觉发布者 (向 C++ 大脑喂图)
    vision_pub = mw_node.create_vision_publisher("VisionDataTopic")
    
    # 2. 注册动作订阅回调 (对齐 Node.h 中的 create_subscription)
    # 这一步会自动配置 C++ 端的 EventLoop 和线程池
    mw_node.create_action_subscription("ActionDataTopic", action_callback)
    
    # 3. 在后台启动 C++ 事件循环 (spin)
    # 使用 daemon=True 确保 Python 退出时线程自动结束
    spin_thread = threading.Thread(target=mw_node.spin, daemon=True)
    spin_thread.start()
    
    frame_counter = 0 
    # ========================================================================

    raw_env = PushT(xml_path=args.env_path, render_mode="rgb_array")
    env = RecordVideo(raw_env, video_folder=args.video_folder, name_prefix="callback_pusht")
    obs, _ = env.reset()

    logger.info("🚀 联调开始：大脑已上线，开始实时推理循环...")

    try:
        while True:
            # 1. 获取画面并泵入 C++
            frame_bgr = cv2.cvtColor(obs["cam_top"], cv2.COLOR_RGB2BGR)
            frame_counter += 1
            
            msg = tinymiddleware_py.VisionMsg()
            msg.frame_id = frame_counter
            msg.width, msg.height, msg.channels = frame_bgr.shape[1], frame_bgr.shape[0], frame_bgr.shape[2]
            msg.payload = frame_bgr.tobytes()
            
            vision_pub.publish(msg)
            
            # 2. 等待动作回调更新缓冲区
            # 这里设置一个超简易的同步机制：每帧发出去后，等待直到最新动作被填入
            start_wait = time.time()
            action_to_exec = None
            
            while action_to_exec is None:
                with action_lock:
                    if latest_action[0] is not None:
                        action_to_exec = latest_action[0]
                        latest_action[0] = None # 消费掉该动作
                
                # 保护性检查：防止大脑宕机导致 Python 死循环
                if time.time() - start_wait > 0.5: # 500ms 超时
                    logger.warning(f"帧 {frame_counter} 推理超时...")
                    break
                time.sleep(0.001) # 1ms 微休眠降低 CPU 占用

            # 3. 执行动作
            if action_to_exec is not None:
                obs, reward, terminated, truncated, info = env.step(action_to_exec)
                
                if frame_counter % 20 == 0:
                    delay = (time.time() - start_wait) * 1000
                    logger.info(f"帧 {frame_counter} | 全链路延迟: {delay:.1f}ms | 坐标: {action_to_exec}")

                if terminated or truncated:
                    break

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
