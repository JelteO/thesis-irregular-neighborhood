# seed.py
# single place to fix all randomness
# Every experiment in this thesis runs on seed 42
# Setting here instead of script menas a run cannot accidentally use
# a different seed somewhere else

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
