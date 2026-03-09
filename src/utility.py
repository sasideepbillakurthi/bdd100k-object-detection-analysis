import torch

# Path to your existing checkpoint
checkpoint_path = 'D:/bdd100k-object-detection-analysis/weights/swin_epoch_1.pth'
output_path = 'D:/bdd100k-object-detection-analysis/weights/model_weights_only.pth'

# 1. Load the full dictionary
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# 2. Pull out only the weights
# This is exactly what your model.load_state_dict() expects
weights_only = checkpoint['model_state_dict']

# 3. Save this as a new file
torch.save(weights_only, output_path)

print(f"✅ Extracted weights saved to: {output_path}")