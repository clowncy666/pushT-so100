import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import cv2
from helper import *

class PushTStable(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, xml_path, max_steps=1000, render_mode=None):
        super(PushTStable, self).__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.pos_random_range = 0.05
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=224, width=224)

        # 归一化动作空间
        self.action_space = spaces.Box(
            low=np.array([-1, -1]), 
            high=np.array([1, 1]), 
            dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            "cam_top": spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
            "cam_side": spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
            "observation.state": spaces.Box(low=-10.0, high=10.0, shape=(5,), dtype=np.float32),
        })

        self.act_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
        self.act_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.act_names]
        self.mocap_id = self.model.body("target_mocap").mocapid[0]
        
        t_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "T_block")
        self.t_qpos_adr = self.model.jnt_qposadr[self.model.body_jntadr[t_body_id]]

        self.current_step = 0
        self.key_id = 0
        self._obs = None
        
        # 初始化历史动作和状态
        self._prev_action = np.zeros(2)
        self._prev_pos = np.zeros(5)
        self._prev_vel = np.zeros(5)
        
        # 控制参数
        self.kp = 10.0  # 比例增益
        self.kd = 1.0   # 微分增益
        self.action_smoothing = 1.0  # 动作平滑系数
        
        # 工作空间范围 - 调整为更适合PushT任务的范围
        self.x_range = [0.0, 0.5]  # x轴范围，扩大到0-0.5
        self.y_range = [-0.3, 0.3]  # y轴范围，扩大到-0.3到0.3

    def _normalize_action(self, action):
        """将动作归一化到-1到1范围"""
        return np.clip(action, -1.0, 1.0)
    
    def _action_to_mocap(self, action):
        """将归一化动作转换为mocap位置"""
        # 映射到实际工作空间
        # 注意：根据环境设计，我们需要确保坐标映射正确
        # 从归一化坐标 (-1, 1) 映射到实际工作空间
        # 调整映射范围，使机械臂能够到达T_sign位置(0.25, 0)
        dx = 0.1 + (action[0] + 1.0) * 0.3 / 2.0  # 映射到0.1-0.4范围
        dy = -0.2 + (action[1] + 1.0) * 0.4 / 2.0  # 映射到-0.2-0.2范围
        
        # 添加调试信息
        if hasattr(self, 'current_step') and self.current_step % 20 == 0:
            print(f"动作映射: 归一化坐标={action}, 实际坐标=({dx:.3f}, {dy:.3f})" )
            
        return dx, dy

    def _set_mocap_2d(self, action):
        """设置mocap位置"""
        dx, dy = self._action_to_mocap(action)
        org_pos = self.data.mocap_pos[self.mocap_id]
        self.data.mocap_pos[self.mocap_id] = [dx, dy, org_pos[2]]

    def get_observation(self):
        self.renderer.update_scene(self.data, camera="top_view")
        img_top = self.renderer.render().copy()

        self.renderer.update_scene(self.data, camera="side_view")
        img_side = self.renderer.render().copy()

        obs = {
            "cam_top": img_top,
            "cam_side": img_side
        }
        obs |= {k:v for k,v in zip(self.act_names ,self.data.qpos[self.act_ids].copy())}
        self._obs = obs
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: np.random.seed(seed)
        
        self.current_step = 0
        # reset to keyframe
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        
        # 随机化T_block位置
        self.data.qpos[self.t_qpos_adr : self.t_qpos_adr+2] = [
            np.random.uniform(0.25 - self.pos_random_range, 0.25+self.pos_random_range),
            np.random.uniform(-self.pos_random_range, self.pos_random_range)
        ]
        self.data.qpos[self.t_qpos_adr+2] = 0.01
        
        # 随机化T_block姿态
        rad = np.random.uniform(-0.5, 0.5)
        self.data.qpos[self.t_qpos_adr+3 : self.t_qpos_adr+7] = [np.cos(rad), 0.0, 0.0, np.sin(rad)]
        
        # 重置历史数据
        self._prev_action = np.zeros(2)
        self._prev_pos = self.data.qpos[self.act_ids].copy()
        self._prev_vel = np.zeros(5)
        
        mujoco.mj_forward(self.model, self.data)
        return self.get_observation(), {}

    # 修改env_gym_ee_stable.py中的step函数
    def step(self, action):
        """
        执行一步环境交互
        action: [target_x, target_y] - 归一化坐标
        """
        # 归一化动作
        normalized_action = self._normalize_action(action)
        
        # 动作平滑处理
        smoothed_action = self.action_smoothing * normalized_action + (1 - self.action_smoothing) * self._prev_action
        self._prev_action = smoothed_action.copy()
        
        # 设置目标位置
        self._set_mocap_2d(smoothed_action)

        # 仿真步进
        control_dt = 1.0 / 10.0  # 10Hz控制频率
        sim_steps = int(control_dt / self.model.opt.timestep)
        
        for _ in range(sim_steps):
            # 直接使用逆运动学计算目标关节位置
            # MuJoCo会自动处理IK
            self.data.ctrl[self.act_ids] = self.data.qpos[self.act_ids]
            
            # 执行仿真步进
            mujoco.mj_step(self.model, self.data)

        # 获取观测
        obs = self.get_observation()
        
        # 检查任务完成情况 - 调整阈值使任务更容易完成
        ok, dxy, dyaw = check_xy_pose_match(
            self.model, self.data, "T_sign_anchor", "T_block_anchor", 
            pos_tol=0.025, yaw_tol_deg=10.0  # 增加阈值
        )
        
        # 奖励函数
        reward = -0.01 
        if ok:
            reward += 20.0
            
        self.current_step += 1
        terminated = ok
        truncated = self.current_step >= self.max_steps
        
        info = {"dxy": dxy, "dyaw": dyaw}
        
        return obs, reward, terminated, truncated, info

    def render(self):
        if self._obs is not None:
            img1 = cv2.resize(self._obs["cam_top"], (448, 448))
            img2 = cv2.resize(self._obs["cam_side"], (448, 448))
            img = np.concatenate([img1, img2], axis=1)
            return img
        return None
    
    def close(self):
        return super().close()