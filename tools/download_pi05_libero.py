from openpi.shared import download
import os

checkpoint_url = "gs://openpi-assets/checkpoints/pi05_libero"
print(f"Downloading {checkpoint_url}...")
checkpoint_dir = download.maybe_download(checkpoint_url)
print(f"Checkpoint downloaded to: {checkpoint_dir}")
