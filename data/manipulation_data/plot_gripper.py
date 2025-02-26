import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

# Define the gripper coordinates
def get_gripper_coords(end_pose,angle, scale=0.01, gripper_state=1):
    # Length of the fingers and base width
    finger_length = 8 * scale
    base_width = 8 * scale * gripper_state
    handle_length = 9 * scale
    
    # Coordinates of the base (rotating)
    base = np.array([
        [-base_width / 2, 0, 0],
        [base_width / 2, 0, 0]
    ])
    
    # Coordinates of the handle (fixed)
    handle = np.array([
        [0, 0, 0],       
        [0, 0, handle_length]
    ])
    
    # Coordinates of the fingers (rotating)
    left_finger = np.array([
        [-base_width / 2, 0, 0],
        [-base_width / 2, 0, -finger_length]
    ])
    
    right_finger = np.array([
        [base_width / 2, 0, 0],
        [base_width / 2, 0, -finger_length]
    ])
    
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

# Create the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.1, bottom=0.25)
ax.set_box_aspect([1, 1, 1])
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])

# Initial angle
end_pose = np.array([0, 0, 0])
initial_angle = 0
scale_factor = 0.3

# Plot the gripper
base, handle, left_finger, right_finger = get_gripper_coords(end_pose,initial_angle, scale=scale_factor)
base_line, = ax.plot(base[:, 0], base[:, 1], base[:, 2], 'k-', lw=2)
handle_line, = ax.plot(handle[:, 0], handle[:, 1], handle[:, 2], 'g-', lw=2)
left_finger_line, = ax.plot(left_finger[:, 0], left_finger[:, 1], left_finger[:, 2], 'b-', lw=2)
right_finger_line, = ax.plot(right_finger[:, 0], right_finger[:, 1], right_finger[:, 2], 'r-', lw=2)

# Add a slider for rotation
ax_angle = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
angle_slider = Slider(ax_angle, 'Angle', -np.pi, np.pi, valinit=initial_angle)

# Add a slider for scale
ax_scale = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
scale_slider = Slider(ax_scale, 'Scale', 0.01, 1, valinit=scale_factor)

# Add a slider for X position
ax_x = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor='lightgoldenrodyellow')
x_slider = Slider(ax_x, 'X', -3, 3, valinit=0)

# Add a gripper state
ax_gripper_state = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor='lightgoldenrodyellow')
gripper_state_slider = Slider(ax_gripper_state, 'Gripper State', 0.05, 1, valinit=1)

# Update function for the slider
def update(val):
    angle = angle_slider.val
    scale_factor = scale_slider.val
    x = x_slider.val
    end_pose = np.array([x, 0, 0])
    gripper_state = gripper_state_slider.val
    base, handle, left_finger, right_finger = get_gripper_coords(end_pose,angle, scale=scale_factor, gripper_state=gripper_state)
    base_line.set_data(base[:, 0], base[:, 1])
    base_line.set_3d_properties(base[:, 2])
    handle_line.set_data(handle[:, 0], handle[:, 1])
    handle_line.set_3d_properties(handle[:, 2])
    left_finger_line.set_data(left_finger[:, 0], left_finger[:, 1])
    left_finger_line.set_3d_properties(left_finger[:, 2])
    right_finger_line.set_data(right_finger[:, 0], right_finger[:, 1])
    right_finger_line.set_3d_properties(right_finger[:, 2])
    fig.canvas.draw_idle()

# Connect the update function to the slider
angle_slider.on_changed(update)
scale_slider.on_changed(update)
x_slider.on_changed(update)
gripper_state_slider.on_changed(update)

plt.show()
