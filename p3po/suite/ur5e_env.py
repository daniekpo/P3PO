import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

import cv2

"""
X: min: -0.2966571473135897 max: 0.49383734924956946
Y: min: -0.80490048532146 max: -0.20859620805567852
Z: min: 0.23781020226938232 max: 0.6900017827785798
"""

def make_homo_transform(R, t):
    R = np.array(R)
    assert R.shape in [(3, 3), (3,), (1, 3)], f"Unexpected R shape {R.shape}"
    if R.ndim == 1:
        R = cv2.Rodrigues(R)[0]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def get_axis_vector_from_matrix(T):
    axis_vector = np.zeros([6])
    axis_vector[:3] = T[:3, 3]
    axis_vector[3:] = cv2.Rodrigues(T[:3, :3])[0].flatten()
    return axis_vector


def pose_to_delta_pose(next_pose, current_pose, return_axis_vector=True):
    """
    Compute the relative (local-frame) delta pose between two absolute poses.

    Args:
        next_pose: 6D array [x, y, z, rx, ry, rz] of the next absolute pose.
        current_pose: 6D array [x, y, z, rx, ry, rz] of the current absolute pose.
        return_axis_vector: if True, returns 6D [x, y, z, rx, ry, rz] delta;
                            otherwise returns 4x4 transformation matrix.

    Returns:
        The local-frame delta transform (either 6D or 4x4).
    """
    T_next = make_homo_transform(next_pose[3:], next_pose[:3])
    T_curr = make_homo_transform(current_pose[3:], current_pose[:3])

    # Compute relative transform in local frame of current pose
    T_delta = np.linalg.inv(T_curr) @ T_next

    if return_axis_vector:
        return get_axis_vector_from_matrix(T_delta)
    return T_delta


def delta_pose_to_pose(delta_pose, current_pose, return_axis_vector=True):
    """
    Args:
        delta_pose: 6d array. Axis aligned delta pose.
        current_pose: 6d array. Axis aligned current robot pose.
        return_axis_vector: whether to returned axis aligned pose (6D) or a
            transformation matrix. Defaults to True

    Returns:
        6D axis aligned pose vector or 4x4 transformation matrix depending.
    """
    delta_T = make_homo_transform(delta_pose[3:], delta_pose[:3])
    current_T = make_homo_transform(current_pose[3:], current_pose[:3])

    new_pose_T = current_T @ delta_T
    if return_axis_vector:
        axis_vector = get_axis_vector_from_matrix(new_pose_T)
        return axis_vector

    return new_pose_T



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

    def __init__(
        self,
        camera_names=None,
        include_depth=True,
        host="169.254.129.1",
        training=True,
        primary_camera_name=None,
        relative_actions=True,
    ):
        super().__init__()
        self.host = host
        self.robot = None
        self.camera_names = camera_names
        self.camera_manager = None
        self.include_depth = include_depth
        self.training = training
        self.primary_camera_name = primary_camera_name or camera_names[0]
        self.robot_home_joints = [
            -4.710016,
            -1.570014,
            1.56999,
            -1.570003,
            4.710003,
            -0.000031,
        ]
        self.relative_actions = relative_actions

        # Action: [x, y, z, rx, ry, rz, gripper]
        # Use conservative bounds for translation/rotation, gripper in [0, 1]
        action_low = np.array(
            [
                -1.0,
                -1.0,
                0.0,  # x, y, z (meters)
                -np.pi,
                -np.pi,
                -np.pi,  # rx, ry, rz (radians)
                0.0,  # gripper
            ],
            dtype=np.float32,
        )
        action_high = np.array(
            [1.0, 1.0, 1.0, np.pi, np.pi, np.pi, 1.0], dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=action_low, high=action_high, dtype=np.float32
        )

        # Observation: images, depths (optional), joints, pose
        obs_spaces = {
            "pixels": spaces.Dict(
                {
                    name: spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8)
                    for name in self.camera_names
                }
            ),
            "joints": spaces.Box(-np.pi, np.pi, shape=(6,), dtype=np.float32),
            "pose": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "features": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
        }

        if self.include_depth:
            obs_spaces["depths"] = spaces.Dict(
                {
                    name: spaces.Box(0, 65535, shape=(480, 640), dtype=np.uint16)
                    for name in self.camera_names
                }
            )
        self.observation_space = spaces.Dict(obs_spaces)

        # We don't need this for training since training is done offline
        if not self.training:
            self.__init_robot__(training=self.training)
            self.__init_camera__()

    def __init_camera__(self):
        from rvkit.camera import CameraManager

        self.camera_manager = CameraManager(camera_names=self.camera_names)

    def __init_robot__(self, training):
        from rvkit.control import Robot

        read_only = training
        self.robot = Robot(host=self.host, read_only=read_only)
        if not training:
            self.robot.movej(self.robot_home_joints)
            self.robot.open_gripper()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if not self.training:
            self.robot.movej(self.robot_home_joints)
            self.robot.open_gripper()

        obs = self._get_obs()
        return obs

    def step(self, action):
        if not self.training:
            if self.relative_actions:
                delta_pose = np.array(action[:6], dtype=np.float32)
                current_pose = np.array(self.robot.getl())
                next_pose = delta_pose_to_pose(delta_pose, current_pose)
            else:
                next_pose = action[:6]

            gripper = action[6]
            self.robot.movel(next_pose, acc=0.1, vel=0.1)

            # Gripper control
            if gripper >= 0.5:
                self.robot.open_gripper()
            else:
                self.robot.close_gripper()

        obs = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = False
        info = {
            "success": True,
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        cam_data = None
        robot_j = None
        robot_l = None

        if self.training:
            cam_data = {
                self.primary_camera_name: (
                    np.zeros((480, 640, 3), dtype=np.uint8),
                    np.zeros((480, 640), dtype=np.uint16),
                ),
            }
            robot_j = np.zeros((6,), dtype=np.float32)
            robot_l = np.zeros((6,), dtype=np.float32)
        else:
            cam_data = self.camera_manager.get_data()
            robot_j = np.array(self.robot.getj(), dtype=np.float32)
            robot_l = np.array(self.robot.getl(), dtype=np.float32)

        obs = {
            "pixels": cam_data[self.primary_camera_name][0],
            "joints": robot_j,
            "pose": robot_l,
            "features": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }
        if self.include_depth:
            # depths = {name: data[name][1] for name in self.camera_names if name in data}
            depth_scale = 0.001  # mm to m
            depth = cam_data[self.primary_camera_name][1] * depth_scale
            obs["depths"] = depth
        return obs

    def render(self, mode="human"):
        obs = self._get_obs()
        img = obs["pixels"]
        return img

    def get_depth(self):
        obs = self._get_obs()
        depth = obs["depths"]
        return depth

    def seed(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._seed = seed
        return [seed]

    def close(self):
        self.robot.close()
