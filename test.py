import torch
print(torch.version.cuda)          # phải khác None
print(torch.cuda.is_available())   # True
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")