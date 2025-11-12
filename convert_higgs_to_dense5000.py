#!/usr/bin/env python3
"""
convert_higgs_to_dense5000.py

Auto-detects input format (MatrixMarket or edge-list) and converts to a dense 5000x5000 matrix.
Usage:
  python3 convert_higgs_to_dense5000.py <input_file> [mode]
mode: pad (default) | crop | tile
Outputs:
  higgs_dense_5000.npy
  higgs_dense_5000.csv  (only if save_csv=True)
"""

import sys
import os
import numpy as np

save_csv = True        # set False to skip CSV (saves time & disk)
target_size = 5000
mode_default = "pad"

if len(sys.argv) < 2:
    print("Usage: python3 convert_higgs_to_dense5000.py <input_file> [mode]")
    sys.exit(1)

fn = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else mode_default
mode = mode.lower()
if mode not in ("pad", "crop", "tile"):
    print("Mode must be one of: pad, crop, tile")
    sys.exit(1)

print("Input:", fn, " Mode:", mode)

# helper to read first non-empty line
def first_nonempty_line(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s != "":
                return s
    return ""

first = first_nonempty_line(fn)
print("First non-empty line:", first[:200])

is_matrix_market = first.startswith("%%MatrixMarket")
if is_matrix_market:
    print("Detected Matrix Market format.")
    # use scipy to read
    try:
        from scipy.io import mmread
    except Exception as e:
        print("ERROR: scipy is required to read Matrix Market files. Install with: pip3 install scipy")
        raise

    sparse = mmread(fn)
    print("Loaded sparse matrix shape:", sparse.shape, " nnz:", getattr(sparse, "nnz", "N/A"))
    dense = sparse.todense()
    A = np.asarray(dense, dtype=np.float32)
else:
    print("Detected edge-list or plain numeric file.")
    # load while skipping lines starting with '%' (common comment char for some datasets)
    # numpy.loadtxt's 'comments' parameter default is '#'; set to '%' so % lines are ignored
    try:
        data = np.loadtxt(fn, comments='%', dtype=np.float64)
    except Exception as e:
        # fallback: try to parse whitespace separated two/three columns manually
        print("numpy.loadtxt failed:", e)
        print("Attempting manual parse (handles irregular whitespace)...")
        cols = []
        with open(fn, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('%') or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                # keep first two tokens (source, target)
                cols.append((float(parts[0]), float(parts[1])))
        if not cols:
            raise RuntimeError("Could not parse input file as edge-list or numeric matrix.")
        data = np.array(cols, dtype=np.float64)

    print("Loaded numeric data shape:", data.shape)
    # If data looks like coordinate format (three columns: i j value), assemble sparse
    if data.ndim == 2 and data.shape[1] >= 3:
        # interpret as coordinate list (i, j, value)
        rows = data[:,0].astype(int)
        cols = data[:,1].astype(int)
        vals = data[:,2].astype(np.float32)
        max_row = int(rows.max())
        max_col = int(cols.max())
        print("Coordinate entries detected. max_row, max_col:", max_row, max_col)
        from scipy.sparse import coo_matrix
        # matrix is 1-indexed in many files (Matrix Market style). detect and adjust if needed
        if rows.min() == 1 or cols.min() == 1:
            rows0 = rows - 1
            cols0 = cols - 1
        else:
            rows0 = rows
            cols0 = cols
        sparse = coo_matrix((vals, (rows0, cols0)), shape=(max_row, max_col))
        dense = sparse.todense()
        A = np.asarray(dense, dtype=np.float32)
    elif data.ndim == 2 and data.shape[1] == 2:
        # treat as edge list (source, target)
        edges = data.astype(int)
        unique_nodes = np.unique(edges)
        print("Edge-list: unique nodes:", len(unique_nodes))
        # choose target_size nodes: sample if more available
        if len(unique_nodes) > target_size:
            selected = np.random.choice(unique_nodes, target_size, replace=False)
        else:
            selected = unique_nodes
        selected = np.sort(selected)
        node_to_idx = {node: i for i, node in enumerate(selected)}
        A = np.zeros((target_size, target_size), dtype=np.float32)
        for src, dst in edges:
            # adjust 1-based to 0-based if necessary (heuristic)
            if src == 0 or dst == 0:
                src0 = src
                dst0 = dst
            else:
                # if min node id == 1 treat as 1-based
                if unique_nodes.min() == 1:
                    src0 = src - 1
                    dst0 = dst - 1
                else:
                    src0 = src
                    dst0 = dst
            if src0 in node_to_idx and dst0 in node_to_idx:
                i = node_to_idx[src0]; j = node_to_idx[dst0]
                A[i, j] = 1.0
                A[j, i] = 1.0
    else:
        # fallback: the numeric data might be a small dense matrix (rows x cols)
        A = np.asarray(data, dtype=np.float32)

print("Dense matrix shape before resize:", A.shape)

# Crop, pad or tile to target_size x target_size
m, n = A.shape
if m == target_size and n == target_size:
    out = A
elif mode == "crop":
    out = A[:target_size, :target_size]
    print("Cropped to", out.shape)
elif mode == "tile":
    reps_row = (target_size + m - 1) // m
    reps_col = (target_size + n - 1) // n
    T = np.tile(A, (reps_row, reps_col))
    out = T[:target_size, :target_size]
    print("Tiled to", out.shape)
else:  # pad
    out = np.zeros((target_size, target_size), dtype=A.dtype)
    out[:m, :n] = A
    print("Padded to", out.shape)

# Save outputs
np.save("higgs_dense_5000.npy", out)
print("Saved higgs_dense_5000.npy (float32)")
if save_csv:
    print("Saving CSV (large)...")
    np.savetxt("higgs_dense_5000.csv", out, delimiter=",")
    print("Saved higgs_dense_5000.csv")

print("Done.")
