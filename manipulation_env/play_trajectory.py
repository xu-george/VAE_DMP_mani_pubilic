"""
replay the record trajectory in the pybullet environment with ur10 
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from manipulation_env.env import ManiEnv
from manipulation_env.robot import UR10Robotiq85
import cv2

cwd = os.getcwd()
model_path = "/manipulation_env/models/urdf/ur10_robotiq_85.urdf"
dict_name = cwd + "/data/manipulation_data"
table_path = "../manipulation_env/models/urdf/objects/table/table.urdf"
block_path = "../manipulation_env/models/YcbPottedMeatCan/model.urdf"

# we already control at the end effector the same as real robot
gripper_offset = -0.03

# load file from dict
trajectories = []
task_ids = []
for file in os.listdir(dict_name):
    if file.endswith(".npy"):
        file_name = os.path.join(dict_name, file)
        
        data = np.load(file_name, allow_pickle=True)
        trajectories.append(data)
        # task id is the first number in the file name
        task_id = int(file.split("_")[0])
        task_ids.append(task_id)   

# load the environment
robot = UR10Robotiq85(model_path=model_path)
env = ManiEnv(robot, block_path=block_path, table_path=table_path, vis=True)
env.reset()

# store images in all the time steps
image_width = 1920
image_height = 1080

for num_traj in range(len(trajectories)):
    # pause by key
    input("Press Enter to continue...")
    # clearn trajectory
    env.clean_traj_plot()
    traj = trajectories[num_traj]
    task_id = task_ids[num_traj]

    images = []
    seg_indexs = []

    env.reset()
    print("task id: ", task_id)
    block_position = np.array([0, 0, 0.04])

    # plot the trajectory
    print("traj shape: ", traj.shape)
    env.traj_plot(traj[:,:3], color = [1, 0, 0])
    # assign the can position
    if task_id == 1:
        block_position[:2] = traj[-1][:2]
        env.reset_block(pos = block_position, euler_angle=[0, 0, 0])
    elif task_id == 2:
        position = traj[20]    # pushing the block at 20 steps
        block_position[0] = position[0] + 0.1
        block_position[1] = position[1]
        env.reset_block(pos = block_position, euler_angle=[0, 0, np.pi/2])
    elif task_id == 3:
        position = traj[20] 
        block_position[0] = position[0]
        block_position[1] = position[1]
        env.reset_block(pos = block_position, euler_angle=[0, 0, 0])

    # replay the trajectory and store the images
    for i in range(len(traj)):
        action = traj[i]
        action[2] += gripper_offset
        env.step(action, control_method='end')    
        env.step_simulation() 
        # # store the image
        # rgb_image, seg_index = env.render(image_width, image_height) 
        # # save nparray to image    
        # seg_indexs.append(seg_index) 
        # images.append(rgb_image)

    # # # pause for a while
    # input("Press Enter to continue...")
    # env.close()

    # # save the images
    # image_path = cwd + "/pictures/images/task_" + str(task_id)
    # if not os.path.exists(image_path):
    #     os.makedirs(image_path)

    # for i in range(len(images)):
    #     image = images[i]
    #     seg_index = seg_indexs[i]
    #     cv2.imwrite(image_path + "/image_" + str(i) + ".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    #     np.save(image_path + "/seg_index_" + str(i), seg_index)

