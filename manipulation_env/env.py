import time
import math
import random

import numpy as np
import pybullet as p
import pybullet_data
from collections import namedtuple
from tqdm import tqdm
from PIL import Image


class Env: 
    
    SIMULATION_STEP_DELAY = 1 / 240.

    def __init__(self, robot, block_path=None, table_path=None, vis=False) -> None:
        """
        robot: robot object
        models: models object

        """
        self.robot = robot
        self.block_path = block_path
        self.table_path = table_path

        self.vis = vis
        if self.vis:
            self.p_bar = tqdm(ncols=0, disable=False)

        # define environment
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        
        self.planeID = p.loadURDF("plane.urdf", [0, 0, -0.72], useFixedBase=True)

        self.robot.load()
        self.robot.step_simulation = self.step_simulation
        
        self.tableID = p.loadURDF(self.table_path, [0.0, 0, -0.74],
                                # p.getQuaternionFromEuler([0, 1.5706453, 0]),
                                p.getQuaternionFromEuler([0, 0, 0]),
                                useFixedBase=True,
                                flags=p.URDF_MERGE_FIXED_LINKS | p.URDF_USE_SELF_COLLISION)
        
        self.boxID = p.loadURDF(self.block_path, [0.6, 0, 0.05],
                                # p.getQuaternionFromEuler([0, 1.5706453, 0]),
                                p.getQuaternionFromEuler([0, 0, 0]),
                                useFixedBase=False,
                                flags=p.URDF_MERGE_FIXED_LINKS | p.URDF_USE_SELF_COLLISION)

    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)
            self.p_bar.update(1)

    def step(self, action, control_method='joint'):
        """
        action: (x, y, z, roll, pitch, yaw, gripper_opening_length) for End Effector Position Control
                (a1, a2, a3, a4, a5, a6, a7, gripper_opening_length) for Joint Position Control
        control_method:  'end' for end effector position control
                         'joint' for joint position control
        """
        assert control_method in ('joint', 'end')
        self.robot.move_ee(action[:-1], control_method)
        self.robot.move_gripper(action[-1])
        for _ in range(20):  # Wait for a few steps
            self.step_simulation()

        reward = 0
        done = True if reward == 1 else False
        info = None
        return reward, done, info

    def reset(self):
        self.robot.reset()

    def close(self):        
        p.disconnect(self.physicsClient)
            

class ManiEnv(Env):
    def __init__(self, robot, block_path=None, table_path=None, vis=False):
        super().__init__(robot, block_path=block_path, table_path=table_path,vis=vis) 

        # define workspace
        self.min_pose = [0.4, -0.4, 0] 
        self.max_pose = [1, 0.4, 0.8]          

    def reset_block(self, pos, euler_angle):
        p.resetBasePositionAndOrientation(self.boxID, pos, p.getQuaternionFromEuler(euler_angle))

    def execute_trajectory(self, trajectory):
        # action = (x, y, z, yaw, gripper_opening_length)
        for action in trajectory:
            self.step(action, control_method='end')
        return self.check_success()

    def check_success(self):
        return True 
    
    def step(self, action, control_method='end'):
        # action: (x, y, z, yaw, gripper_opening_length)    

        gripper_width = self.robot.get_gripper_width()
        # update position
        pose = action[:3]
        # limit the position
        pose = np.clip(pose, self.min_pose, self.max_pose)
        # update orientation
        rotation = np.array([0, np.pi/2, action[3]])    
      
        # update gripper        
        gripper_width = action[4] * (self.robot.gripper_range[1] - self.robot.gripper_range[0])  + self.robot.gripper_range[0]
        
        self.robot.move_ee(np.concatenate([pose, rotation]), control_method)
        self.robot.move_gripper(gripper_width)

        for _ in range(120):  # Wait for a few steps
            self.step_simulation()  

    def get_block_position(self):
        return p.getBasePositionAndOrientation(self.boxID)[0]

    def render(self, width=1920, height=1080):
        """
        Renders a high-quality image from a PyBullet simulation and saves it to a file.

        Parameters:
        - filename: str, the path to the file where the image will be saved.
        - width: int, the width of the rendered image.
        - height: int, the height of the rendered image.
        """
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 1)

        # Define camera parameters
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0],
            distance=2.0,
            yaw=44.4,
            pitch=-32.0,
            roll=0,
            upAxisIndex=2
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(width) / height,
            nearVal=0.1,
            farVal=100.0
        )

        # Capture the image
        _, _, img, _, seg_index = p.getCameraImage(
            width,
            height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,        
        )
        return img, seg_index
    
    def traj_plot(self, traj, color=None):
        """
        plot the trajectory
        """
        if color is None:
            # generate a random
            color = np.random.rand(3)
                        
        for i in range(len(traj) - 1):
            start_point = traj[i]
            end_point = traj[i + 1]
            p.addUserDebugLine(start_point, end_point, color, 4)

        # add the last point
        p.addUserDebugLine(traj[1], traj[1], [0, 0, 1], lineWidth=10)
        p.addUserDebugLine(traj[-1], traj[-1], [1, 0, 0], lineWidth=10)

    def plot_dot(self, pos, text, color=None):
        """
        plot a dot
        """
        if color is None:
            color = np.random.rand(3)
        p.addUserDebugLine(pos, pos, color, lineWidth=50)
        # add text
        p.addUserDebugText(text, pos, textColorRGB=[0, 0, 0], textSize=1)

    def clean_traj_plot(self):
        """
        clean the trajectory plot
        """
        p.removeAllUserDebugItems()

    def check_touching(self):
        """
        check if the robot gripper is touching the block
        """        
        contact_points = p.getContactPoints(self.robot.id, self.boxID)
        return len(contact_points) > 0
    
    def check_grasping(self):
        """
        check if both robot gripper touch the block
        """
        finger_id_a = 12
        finger_id_b = 17

        left_contact = p.getContactPoints(self.robot.id, self.boxID, finger_id_a)
        right_contact = p.getContactPoints(self.robot.id, self.boxID, finger_id_b)    
        grasped = (left_contact != () or right_contact != ())
        return grasped
    
    def check_arrived(self, target_pos, tol=0.01):
        """
        check if the block has arrived at the target position
        """
        ee_pos = self.get_block_position()
        return abs(ee_pos[0] - target_pos[0]) < tol   
