import numpy as np
import math

### TASK 1a)
M = 4
N = 3
associations = np.min([M, N])

num_events = 0
for m in range(associations):
    num_events += (math.factorial(N) * math.factorial(M)) / (math.factorial(m) * math.factorial(N - m) * math.factorial(M - m))

print(f"Task 1a solution: {num_events}")

### TASK 2a)

gated_ratio = 17/49

print(f"Task 1b solution: {gated_ratio}")

### TASK 2d)

num_events = 0
for m in range(associations):
    num_events += (2**(M-m) * math.factorial(N) * math.factorial(M)) / (math.factorial(m) * math.factorial(N - m) * math.factorial(M - m))
    
print(f"Task 1d solution: {num_events}")