import torch
try:
    data = torch.load("calibration_data.pt", weights_only=False)
    print(f"Loaded {len(data)} samples")
    sample = data[0]
    print(f"Sample type: {type(sample)}")
    if isinstance(sample, (tuple, list)):
        print(f"Sample length: {len(sample)}")
        for i, item in enumerate(sample):
            print(f"Item {i}: {type(item)} {item.shape if hasattr(item, 'shape') else ''}")
    else:
        print(f"Sample: {sample}")
except Exception as e:
    print(f"Error: {e}")
