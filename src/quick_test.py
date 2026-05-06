import torch
checkpoint = torch.load("best_model_backbone_37-03.pt", map_location="cpu")
# If it's a Lightning/Hydra checkpoint, it might be nested
state_dict = checkpoint.get('model_state_dict', checkpoint)
print("First 10 keys in your checkpoint:")
for k in list(state_dict.keys())[:10]:
    print(k)