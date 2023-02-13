import sys
sys.path.append("./python")
import numpy as np
import needle as ndl
import needle.nn as nn

sys.path.append("./apps")

print(ndl.init.xavier_uniform(3, 5, gain=1.5).numpy())