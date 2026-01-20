import os
import subprocess
import numpy as np
import time

input_pipe = "input_pipe"
output_pipe = "output_pipe"

# Generate dummy input
# Model inputs:
# obs: (1, 3, 224, 224)? No, let's check export.
# I need to know input names and shapes.
# From export_onnx.py or model inspection.
# Assuming standard Pi0 inputs.

# For verification, I'll essentially run trtexec in background reading from pipe,
# and writing to Python.
# But trtexec might not support keeping session open (persistent mode).
# It does support --noDataTransfer --useSpinWait etc.
# But standard trtexec is one-shot or benchmark loop.
# It doesn't run as a server.

# If trtexec is one-shot, reloading engine every time is SLOW (seconds).
# I need persistent runner.

# If I cannot use trtexec as a server, this wrapper is too slow (model load time).
pass
