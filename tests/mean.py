from collections import deque
import numpy as np

a = deque(maxlen=10)
for idx in range(100):
    a.append(idx)
    print(len(a))

print(a)
print(np.mean(a))