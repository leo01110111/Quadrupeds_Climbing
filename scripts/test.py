import torch
num_envs = 5
history = 3
a = torch.randn((num_envs, history, 4, 3))
print(a)
a_1 = torch.norm(a[:,:,:], dim=-1)
print("a1:", a_1)
a_1 = torch.norm(a[:,:,:,:], dim=-1)
print("a1d:", a_1)
a_2 = torch.max(a_1, dim=1)
print("a_2:",a_2)
a_3 = a_2[0]
print("a_3:",a_3)
a_3 = a_3.unsqueeze(1)
print("Unsq:", a_3)
a_4 = torch.any(a_3 > 2.0, dim=1)
print("a_4",a_4)
print("a_4 shape",a_4.shape)
foot_contact_states = torch.any(
        torch.max(torch.norm(a[:,:,:], dim=-1), dim=1)[0].unsqueeze(1) > 1.0, dim=1 
    )
print("foot_contact_states",foot_contact_states.shape)