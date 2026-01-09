import numpy as np
import robosuite.utils.transform_utils as T
import math

def my_quat2axisangle(quat):
    # My fixed version (assuming wxyz)
    if quat[0] > 1.0: quat[0] = 1.0
    elif quat[0] < -1.0: quat[0] = -1.0
    den = np.sqrt(1.0 - quat[0] * quat[0])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (quat[1:] * 2.0 * math.acos(quat[0])) / den

def old_quat2axisangle(quat):
    # Old buggy version (assuming xyzw?)
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

# Test Identity Quaternion (w=1, x=0, y=0, z=0)
q_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
print(f"Testing Identity [1, 0, 0, 0] (as wxyz):")
print(f"  Official T.quat2axisangle: {T.quat2axisangle(q_wxyz)}")
print(f"  My Fixed Function:         {my_quat2axisangle(q_wxyz)}")
print(f"  Old Buggy Function:        {old_quat2axisangle(q_wxyz)}")

# Test 90 deg rotation around X
# Axis: [1, 0, 0], Angle: pi/2
# Quat: [cos(pi/4), sin(pi/4)*1, 0, 0] = [0.707, 0.707, 0, 0]
q_rot_x = np.array([0.7071068, 0.7071068, 0.0, 0.0])
print(f"\nTesting 90 deg X-axis [0.707, 0.707, 0, 0] (as wxyz):")
print(f"  Official T.quat2axisangle: {T.quat2axisangle(q_rot_x)}")
print(f"  My Fixed Function:         {my_quat2axisangle(q_rot_x)}")
print(f"  Old Buggy Function:        {old_quat2axisangle(q_rot_x)}")
