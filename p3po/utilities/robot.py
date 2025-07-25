import time

import rtde_control
import rtde_receive
import numpy as np

from utils.robot_gripper import RobotiqGripper
from utils.constants import home_j

default_home_joint_pose = [
    -5.025327865277426,
    -1.4772643607905884,
    0.8192513624774378,
    -0.9127100271037598,
    4.7187347412109375,
    -0.3162348906146448,
]

'''
Note: The z bounds and offset are based on the gripper. This works for the case where the
the camera is not mounted on the gripper. If you're using the camera mounted on the
gripper, you need to adjust the z offset to 0 and set the correct z bounds.
'''

x_bounds = [-0.2924608360264403, 0.2659200928056559]
y_bounds = [-0.6660059850909836, -0.34361393313539224]
z_bounds = [0.07379502840676555, 0.5992787597081022]

default_tcp_bounds = [x_bounds, y_bounds, z_bounds]

class Robot:
    def __init__(self, host='169.254.129.1', home_joints=None, read_only=False, tcp_bounds=None, tcp_offset_z=0.1734126403712062):
        self.rtde_c = None
        self.rtde_r = None
        self.host = host
        self.read_only = read_only
        self.home_joints = home_joints or home_j
        self.gripper = None
        self.tcp_offset_z = tcp_offset_z

        self.tcp_bounds = tcp_bounds or default_tcp_bounds
        self.connect()

    def clip_pose(self, pose):
        if self.tcp_bounds is not None:
            # pose[0] = np.clip(pose[0], self.tcp_bounds[0][0], self.tcp_bounds[0][1])
            # pose[1] = np.clip(pose[1], self.tcp_bounds[1][0], self.tcp_bounds[1][1])
            pose[2] = np.clip(pose[2], self.tcp_bounds[2][0], self.tcp_bounds[2][1])
        return pose

    def connect(self):
        if self.rtde_r is None or not self.rtde_r.isConnected():
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.host)

        if not self.read_only:
            if self.rtde_c is None or not self.rtde_c.isConnected():
                self.rtde_c = rtde_control.RTDEControlInterface(self.host)
                self.set_tcp_z_offset(self.tcp_offset_z)

                print(f"TCP offset set to {self.tcp_offset_z}. This is based on the z offset of the gripper. Please check if this is correct.")
    
    def set_tcp_z_offset(self, z_offset):
        current_offset = self.rtde_c.getTCPOffset()
        current_offset[2] = z_offset
        self.rtde_c.setTcp(current_offset)

    def get_tcp_offset(self):
        return self.rtde_c.getTCPOffset()


    def __connect_gripper(self):
        if self.gripper is None:
            self.gripper = RobotiqGripper()
            self.gripper.connect(self.host, 63352)
            self.gripper.activate()


    def reconnect(self, read_only=False):
        self.disconnect()
        self.read_only = read_only

        if self.rtde_r is not None:
            self.rtde_r.reconnect()

        if not self.read_only:
            if self.rtde_c is not None:
                self.rtde_c.reconnect()
            else:
                self.rtde_c = rtde_control.RTDEControlInterface(self.host)


    def disconnect(self):
        if self.rtde_c is not None:
            self.rtde_c.disconnect()
        if self.rtde_r is not None:
            self.rtde_r.disconnect()
        if self.gripper is not None:
            self.gripper.disconnect()

    def go_home(self, acc=0.2, vel=0.2, random_mutation=False, asynchronous=False):
        if self.home_joints is None:
            raise ValueError("Home pose not set")
        if random_mutation:
            pose = self.home_joints.copy()
            # randomly add or subtract 0.01 to 0.03 to the x,y,z of the home pose
            pose[0] += np.random.uniform(-0.03, 0.03)
            pose[1] += np.random.uniform(-0.03, 0.03)
            pose[2] += np.random.uniform(-0.03, 0.03)
        else:
            pose = self.home_joints
        self.movej(pose, acc, vel, asynchronous)


    def is_moving(self):
        return not self.rtde_c.isSteady()

    def open_gripper(self):
        if self.read_only:
            raise ValueError("Cannot move robot in read-only mode")

        if self.gripper is None:
            self.__connect_gripper()

        self.gripper.open()

    def close_gripper(self):
        if self.read_only:
            raise ValueError("Cannot move robot in read-only mode")

        if self.gripper is None:
            self.__connect_gripper()

        self.gripper.close()

    def movej(self, pose, acc=0.1, vel=0.1, asynchronous=False):
        if self.read_only:
            raise ValueError("Cannot move robot in read-only mode")
        self.rtde_c.moveJ(pose, vel, acc, asynchronous)

    def movel(self, pose, acc=0.1, vel=0.1, asynchronous=False, clip=False):
        if self.read_only:
            raise ValueError("Cannot move robot in read-only mode")
        if clip:
            pose = self.clip_pose(pose)
        self.rtde_c.moveL(pose, vel, acc, asynchronous)

    def movep(self, pose, acc=0.1, vel=0.1, asynchronous=False, clip=False):
        if self.read_only:
            raise ValueError("Cannot move robot in read-only mode")
        if clip:
            pose = self.clip_pose(pose)
        self.rtde_c.movePath(pose, vel, acc, asynchronous)

    def stopl(self):
        if self.read_only:
            raise ValueError("Cannot move robot in read-only mode")
        self.rtde_c.stopL()

    def stopj(self):
        if self.read_only:
            raise ValueError("Cannot move robot in read-only mode")
        self.rtde_c.stopJ()

    def getl(self, include_timestamp=False):
        pose = self.rtde_r.getActualTCPPose()
        if include_timestamp:
            return pose, time.time()
        return pose

    def getj(self, include_timestamp=False):
        joints = self.rtde_r.getActualQ()
        if include_timestamp:
            return joints, time.time()
        return joints

    def close(self):
        self.disconnect()