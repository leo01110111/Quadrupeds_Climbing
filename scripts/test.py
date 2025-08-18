import torch

num_envs = 5
vel_command_b = torch.zeros((num_envs, 3))
target_vec_b = torch.ones((num_envs, 3))
velocity = torch.ones(num_envs,1) 
vel_command_b[:, :2] = target_vec_b[:,:2] * velocity #(5,2) * (5,1) <- gets broadcasted
print(vel_command_b)