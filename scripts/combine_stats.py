import json

# Load Original (Safe Std)
with open("torch_norm_stats.json.bak", "r") as f:
    orig = json.load(f)

# Load Calibrated (Correct Mean, Too narrow Std)
with open("torch_norm_stats.json", "r") as f:
    calib = json.load(f)

# Combine: Calibrated Mean + Original Std
# For State
orig["state"]["mean"] = calib["state"]["mean"]
# orig["state"]["std"] keep as is

print("Combined Stats:")
print("Mean (from Calibrated):", orig["state"]["mean"])
print("Std (from Original):", orig["state"]["std"])

# Save
with open("torch_norm_stats.json", "w") as f:
    json.dump(orig, f, indent=2)
