import gymnasium as gym
from gymnasium import spaces
import numpy as np

from utils.robot import Robot
from camera_manager import CameraManager

class UR5Env(gym.Env):
    """
    Gymnasium environment for UR5e robot with Robotiq gripper and RealSense cameras.
    Action: 7D array [x, y, z, rx, ry, rz, gripper_state]
        - gripper_state: 1=open, 0=closed
    Observation: dict with
        - 'images': dict of {camera_name: rgb image}
        - 'depths': dict of {camera_name: depth image} (optional)
        - 'joints': robot joint angles
        - 'pose': robot TCP pose
    """
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, camera_names=None, include_depth=False, host='169.254.129.1'):
        super().__init__()
        self.robot = Robot(host=host)
        self.camera_manager = CameraManager(camera_names=camera_names)
        self.include_depth = include_depth

        # Action: [x, y, z, rx, ry, rz, gripper]
        # Use conservative bounds for translation/rotation, gripper in [0, 1]
        action_low = np.array([
            -1.0, -1.0, 0.0,   # x, y, z (meters)
            -np.pi, -np.pi, -np.pi,  # rx, ry, rz (radians)
            0.0  # gripper
        ], dtype=np.float32)
        action_high = np.array([
            1.0, 1.0, 1.0,
            np.pi, np.pi, np.pi,
            1.0
        ], dtype=np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        # Observation: images, depths (optional), joints, pose
        obs_spaces = {
            'images': spaces.Dict({
                name: spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8) for name in self.camera_names}),
            'joints': spaces.Box(-np.pi, np.pi, shape=(6,), dtype=np.float32),
            'pose': spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
        }

        if self.include_depth:
            obs_spaces['depths'] = spaces.Dict({name: spaces.Box(0, 65535, shape=(480, 640), dtype=np.uint16) for name in self.camera_names})
        self.observation_space = spaces.Dict(obs_spaces)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.robot.go_home()
        self.robot.open_gripper()

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        pose = np.array(action[:6], dtype=np.float32)
        gripper = action[6]
        self.robot.movel(pose, acc=0.1, vel=0.1)

        # Gripper control
        if gripper >= 0.5:
            self.robot.open_gripper()
        else:
            self.robot.close_gripper()

        obs = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        data = self.camera_manager.get_data()
        images = {name: data[name][0] for name in self.camera_names if name in data}
        obs = {
            'images': images,
            'joints': np.array(self.robot.getj(), dtype=np.float32),
            'pose': np.array(self.robot.getl(), dtype=np.float32),
        }
        if self.include_depth:
            depths = {name: data[name][1] for name in self.camera_names if name in data}
            obs['depths'] = depths
        return obs

    def render(self, mode="human"):
        # Simple render: show first camera's RGB image
        import cv2
        obs = self._get_obs()
        img = next(iter(obs['images'].values()))
        cv2.imshow("UR5Env Camera", img)
        cv2.waitKey(1)

    def close(self):
        self.robot.close()