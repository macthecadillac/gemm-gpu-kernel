import time
import numpy as np


n = 4096
a = np.full((n, n), 2.)
b = np.full((n, n), 3.)
c = np.full((n, n), 0.1)
alpha = 1.
beta = 1.

start = time.time()
c = alpha * a * b + beta * c
end = time.time()
print(end - start)
