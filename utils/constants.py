import numpy as np

SCALE = np.array([1, 1, 1], dtype=np.float32)
OFFSET = np.array([0, 0, 0], dtype=np.float32)

# workspace limits
TRANS_MIN = np.array([0, -0.5, 0])
TRANS_MAX = np.array([1.0, 0.5, 0.3])

# gripper limits
WIDTH_MAX = 0.11   # meters

# use rot6d as continuous rotation representation
# ref: [On the Continuity of Rotation Representations in Neural Networks](http://arxiv.org/abs/1812.07035)
ACTION_DIM = 10     # (xyz, rot6d, width)