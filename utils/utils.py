import numpy as np
import json
import math

# common use files in the project
# calcuate yaw rotation
def quaternion_to_yaw(w, x, y, z):
    # Calculate the yaw (Z-axis rotation)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    # Convert the yaw from radians to degrees
    yaw_degrees = math.degrees(yaw)
    return yaw_degrees

# a simple function to remove the position that are too close
def remove_close_position(position, threshold = 0.02):
    new_position = []
    save_index = []
    new_position.append(position[0])
    save_index.append(0)
    for i in range(1, len(position)):
        if np.linalg.norm(np.array(position[i]) - np.array(new_position[-1])) > threshold:
            new_position.append(position[i])
            save_index.append(i)
    return np.array(new_position), save_index

# smooth the position data and keep the final value the same
def smooth_position(position, window_size = 5):
    new_position = []
    for i in range(len(position) - window_size):
        new_position.append(np.mean(position[i:i+window_size], axis = 0))
    new_position.append(position[-1])
    return np.array(new_position)

# gripper plot
def get_gripper_coords(end_pose, angle, scale=0.01, gripper_state=1):
    # Length of the fingers and base width
    finger_length = 8 * scale
    base_width = 2 * scale * gripper_state
    handle_length = 9 * scale
    
    # Coordinates of the base (rotating)
    base = np.array([
        [-base_width / 2, 0, 0],
        [base_width / 2, 0, 0]
    ]) - np.array([0,0,handle_length])
    
    # Coordinates of the handle (fixed)
    handle = np.array([
        [0, 0, 0],       
        [0, 0, -handle_length]
    ])
    
    # Coordinates of the fingers (rotating)
    left_finger = np.array([
        [-base_width / 2, 0, 0],
        [-base_width / 2, 0, -finger_length]
    ]) - np.array([0,0,handle_length])
    
    right_finger = np.array([
        [base_width / 2, 0, 0],
        [base_width / 2, 0, -finger_length]
    ]) - np.array([0,0,handle_length])
    
    # Rotation matrix around Z-axis
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    
    # Rotate the fingers, and base    
    base = np.dot(base, rotation_matrix) + end_pose
    left_finger = np.dot(left_finger, rotation_matrix) + end_pose
    right_finger = np.dot(right_finger, rotation_matrix) + end_pose
    handle += end_pose
    
    return base, handle, left_finger, right_finger

