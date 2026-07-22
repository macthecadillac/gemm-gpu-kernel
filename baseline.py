import time
import numpy as np


n = 4096
a = np.arange(0, 4096 * 4096).reshape(4096, -1) / (2048 * 2048)
b = np.arange(4096 * 4096 - 1, -1, -1).reshape(4096, -1) / (2048 * 2048) - 0.5
c = np.full((n, n), 0.1)
alpha = 1.
beta = 1.

start = time.time()
c = alpha * a @ b + beta * c
end = time.time()
print(end - start)
print(c[0, 0])
print(c[345, 879])
