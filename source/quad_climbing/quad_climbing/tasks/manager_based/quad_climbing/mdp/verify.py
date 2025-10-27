import torch
from omni.isaac.core.utils.torch.rotations import yaw_quat, quat_rotate_inverse, wrap_to_pi

# --------------------------
# USER INPUTS
# --------------------------

# Robot state (world frame)
robot_pos_w = torch.tensor([[2.0, 1.0, 0.0]])      # (x, y, z)
robot_yaw_deg = 90                                 # facing +y direction
robot_heading_w = torch.tensor([robot_yaw_deg * 3.14159 / 180.0]).unsqueeze(0)

# Convert yaw to quaternion
robot_quat_w = yaw_quat(torch.tensor([robot_heading_w[0, 0]]))

# Goal position (world frame)
goal_pos_w = torch.tensor([[5.0, 3.0, 0.0]])       # target x,y,z

# Desired speed
velocity_command = torch.tensor([[1.5]])           # scalar speed

# --------------------------
# COMPUTE TARGET DIRECTION
# --------------------------

# Vector to target in world frame
target_vec_w = goal_pos_w[:, :3] - robot_pos_w[:, :3]
print("Target vector in world frame:", target_vec_w)

# Convert to body frame
target_vec_b = quat_rotate_inverse(robot_quat_w, target_vec_w)
print("Target vector in body frame (unnormalized):", target_vec_b)

# Normalize horizontal direction
norm = torch.norm(target_vec_b[:, :2], dim=1).unsqueeze(1)
target_vec_b_xy = target_vec_b[:, :2] / torch.clamp(norm, min=0.001)
print("Normalized target direction (body x,y):", target_vec_b_xy)

# Linear velocity command (in body frame)
vel_command_b_xy = target_vec_b_xy * velocity_command
print("Linear velocity command (v_x, v_y):", vel_command_b_xy)

# Compute target heading from body-frame target vector
target_heading_w = torch.atan2(target_vec_b_xy[:, 1], target_vec_b_xy[:, 0])
print("Target heading (world frame):", target_heading_w)

# Heading error
target_heading_error_b = wrap_to_pi(target_heading_w - robot_heading_w.squeeze(1))
print("Heading error (body frame):", target_heading_error_b)
