#!/usr/bin/env python3
"""
extract_submatrix_5000.py

Extract a 5000x5000 dense submatrix from a large sparse .mtx without densifying the full matrix.

Usage:
  python3 extract_submatrix_5000.py <mtx_file> [mode]

mode:
  random     - random sample 5000 unique nodes (default)
  topdeg     - choose 5000 nodes with highest degree (denser submatrix)
  seed       - local BFS/ego-net expansion around highest-degree node

Outputs:
  sub_5000.npy         : float32 dense (5000x5000)
  sub_5000.csv         : text CSV (optional)

Note: requires scipy and numpy
"""
import sys
import numpy as np
from scipy.io import mmread
from scipy import sparse

if len(sys.argv) < 2:
    print("Usage: python3 extract_submatrix_5000.py <mtx_file> [random|topdeg|seed]")
    sys.exit(1)

mtx_path = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else "random"
target = 5000
save_csv = True

print("Loading sparse matrix (as sparse)...")
S = mmread(mtx_path)  # returns sparse matrix (coo or other)
print("Type:", type(S), " shape:", S.shape, " nnz:", getattr(S, "nnz", "N/A"))

# ensure CSR for fast row/col slicing and degree computation
if not sparse.isspmatrix_csr(S):
    S = S.tocsr()

nrows, ncols = S.shape
assert nrows == ncols, "Expected square adjacency-like matrix, got shape {}".format(S.shape)

print("Selecting nodes mode:", mode)

if mode == "topdeg":
    # compute degree (sum of absolute values per row)
    deg = np.array(S.getnnz(axis=1))  # integer degrees (nnz per row)
    # find indices of top degrees
    idx = np.argsort(deg)[::-1][:target]
    idx = np.sort(idx)
elif mode == "seed":
    # find highest-degree node and grow BFS/ego net until we have 'target' nodes
    deg = np.array(S.getnnz(axis=1))
    seed = int(np.argmax(deg))
    print("Seed node (highest degree):", seed)
    selected = set([seed])
    frontier = set([seed])
    while len(selected) < target and frontier:
        new_frontier = set()
        for u in frontier:
            # neighbors (from CSR) - find indices of nonzeros in row u
            row_start = S.indptr[u]
            row_end = S.indptr[u+1]
            neighbors = S.indices[row_start:row_end]
            for v in neighbors:
                if v not in selected:
                    new_frontier.add(int(v))
        # add new_frontier to selected
        for v in new_frontier:
            if len(selected) >= target:
                break
            selected.add(v)
        frontier = new_frontier
        # if BFS stalls, add top-degree remaining nodes
        if not frontier and len(selected) < target:
            deg = np.array(S.getnnz(axis=1))
            for v in np.argsort(deg)[::-1]:
                if v not in selected:
                    selected.add(int(v))
                if len(selected) >= target:
                    break
    idx = np.array(sorted(list(selected)))[:target]
    if len(idx) < target:
        # pad with top-deg nodes
        deg = np.array(S.getnnz(axis=1))
        for v in np.argsort(deg)[::-1]:
            if v not in selected:
                idx = np.append(idx, v)
            if len(idx) >= target:
                break
    idx = np.sort(idx.astype(int))
else:
    # random sampling of unique node ids
    rng = np.random.default_rng(seed=12345)
    if nrows <= target:
        idx = np.arange(nrows)
    else:
        idx = rng.choice(nrows, size=target, replace=False)
    idx = np.sort(idx)

print("Selected node count:", len(idx), "min,max:", idx.min(), idx.max())

# Extract submatrix: S[idx, :][:, idx]
print("Extracting submatrix (sparse slicing)...")
S_sub = S[idx, :][:, idx]  # still sparse (CSR->CSR)
print("Submatrix shape:", S_sub.shape, "nnz:", S_sub.nnz)

# Convert small sparse to dense float32
print("Converting to dense (will be ~95 MB as float32)...")
A_dense = S_sub.toarray().astype(np.float32)

# Save files
np.save("sub_5000.npy", A_dense)
print("Saved sub_5000.npy")
if save_csv:
    print("Saving sub_5000.csv (text; may be large)...")
    np.savetxt("sub_5000.csv", A_dense, delimiter=",")
    print("Saved sub_5000.csv")

print("Done.")
