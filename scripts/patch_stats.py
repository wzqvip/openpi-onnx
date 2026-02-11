import json
import numpy as np

with open("torch_norm_stats.json", "r") as f:
    stats = json.load(f)

# Values from Libero Spatial InitState (from logs)
# first10=[-2.11241627e-01 -1.10906019e-02  1.17424952e+00 ...]
new_pos_mean = [-0.211241627, -0.0110906019, 1.17424952]

print("Old Mean:", stats["state"]["mean"])

# Patch first 3 dimensions (Position)
for i in range(3):
    stats["state"]["mean"][i] = new_pos_mean[i]

print("New Mean:", stats["state"]["mean"])

with open("torch_norm_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
