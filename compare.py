import numpy as np
import sys

file1 = sys.argv[1]
file2 = sys.argv[2]

A = np.loadtxt(file1, delimiter=",")
B = np.loadtxt(file2, delimiter=",")

diff = A - B
max_abs = np.max(np.abs(diff))
max_rel = np.max(np.abs(diff) / (np.abs(A) + 1e-12))
mismatch = np.sum(np.abs(diff) > 1e-12)

print("Max Abs Error:", max_abs)
print("Max Rel Error:", max_rel)
print("Mismatches:", mismatch)
