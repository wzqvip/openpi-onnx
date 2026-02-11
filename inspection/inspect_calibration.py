
import torch
import sys

try:
    data = torch.load("calibration_data.pt", weights_only=False)
    print(f"Loaded {len(data)} samples.")
    if len(data) > 0:
        sample = data[0]
        print("Keys:", sample.keys())
        if "state" in sample:
            print("State shape:", sample["state"].shape)
            print("State content preview:", sample["state"])
        if "prompt" in sample:
            print("Prompt:", sample["prompt"])
except Exception as e:
    print(f"Error loading: {e}")
