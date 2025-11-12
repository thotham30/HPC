#!/usr/bin/env python3
"""
Convert a Matrix Market (.mtx) sparse matrix to a dense 5000x5000 matrix.
- Crops or pads the matrix if dimensions differ.
- Saves both .npy (fast) and .csv (for C code).
"""

import numpy as np
from scipy.io import mmread
import sys

# ---- configuration ----
filename = sys.argv[1] if len(sys.argv) > 1 else "higgs-twitter.mtx"
target_size = 5000
save_csv = True   # set False if you only want .npy output

print(f"Loading Matrix Market file: {filename}")
A = mmread(filename)
print("Loaded sparse matrix with shape:", A.shape, "and", A.nnz, "nonzero elements")

# Convert to dense
A = A.todense()
A = np.asarray(A, dtype=np.float32)
print("Converted to dense, dtype:", A.dtype, "shape:", A.shape)

# Crop or pad to 5000x5000
m, n = A.shape
if m >= target_size and n >= target_size:
    A = A[:target_size, :target_size]
    print(f"Cropped to {A.shape}")
else:
    B = np.zeros((target_size, target_size), dtype=A.dtype)
    B[:m, :n] = A
    A = B
    print(f"Padded to {A.shape}")

# Save .npy for fast reload
np.save("higgs_dense_5000.npy", A)
print("✅ Saved binary file: higgs_dense_5000.npy")

# Optionally save CSV (very large!)
if save_csv:
    print("Saving CSV (this can take a few minutes)...")
    np.savetxt("higgs_dense_5000.csv", A, delimiter=",")
    print("✅ Saved CSV file: higgs_dense_5000.csv")

print("Done.")
