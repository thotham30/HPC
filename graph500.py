#!/usr/bin/env python3
"""
Graph500 Kronecker Graph Generator
Generates adjacency matrices compatible with bcv_svd.c
"""

import numpy as np
import sys
import time

def generate_kronecker_edges(scale, edgefactor=16, seed=None):
    """
    Generate Kronecker graph edges following Graph500 specification
    
    Parameters:
    - scale: Graph size is 2^scale vertices
    - edgefactor: Number of edges per vertex (default 16 for Graph500)
    - seed: Random seed for reproducibility
    """
    # Graph500 Kronecker parameters
    A, B, C = 0.57, 0.19, 0.19
    # D = 1 - A - B - C = 0.05
    
    n_vertices = 1 << scale  # 2^scale
    n_edges = n_vertices * edgefactor
    
    print(f"Generating Kronecker graph:")
    print(f"  SCALE = {scale}")
    print(f"  Vertices = {n_vertices:,}")
    print(f"  Edges = {n_edges:,}")
    print(f"  Edge factor = {edgefactor}")
    
    if seed is not None:
        np.random.seed(seed)
    
    edges = []
    
    for i in range(n_edges):
        if i % 100000 == 0:
            print(f"  Generated {i:,} / {n_edges:,} edges...", end='\r')
        
        src, dst = 0, 0
        
        for bit in range(scale):
            r1, r2 = np.random.random(), np.random.random()
            
            # Determine bit values based on Kronecker parameters
            if r1 > A:
                if r1 <= A + B:
                    src_bit = 0
                else:
                    src_bit = 1
            else:
                src_bit = 0
            
            if r2 > A:
                if r2 <= A + C:
                    dst_bit = 0
                else:
                    dst_bit = 1
            else:
                dst_bit = 0
            
            src |= (src_bit << bit)
            dst |= (dst_bit << bit)
        
        edges.append((src, dst))
    
    print(f"  Generated {n_edges:,} / {n_edges:,} edges... Done!")
    return edges, n_vertices


def edges_to_adjacency_matrix(edges, n_vertices, symmetric=True):
    """Convert edge list to dense adjacency matrix"""
    print(f"\nBuilding {n_vertices}x{n_vertices} adjacency matrix...")
    A = np.zeros((n_vertices, n_vertices), dtype=np.float64)
    
    for i, (src, dst) in enumerate(edges):
        if i % 100000 == 0:
            print(f"  Processing edge {i:,} / {len(edges):,}...", end='\r')
        
        A[src, dst] = 1.0
        if symmetric and src != dst:
            A[dst, src] = 1.0
    
    print(f"  Processing edge {len(edges):,} / {len(edges):,}... Done!")
    return A


def save_matrix_csv(A, filename):
    """Save matrix to CSV file"""
    print(f"\nSaving matrix to {filename}...")
    np.savetxt(filename, A, delimiter=',', fmt='%.1f')
    print(f"Saved successfully!")


def save_edge_list(edges, filename):
    """Save edge list to text file"""
    print(f"\nSaving edge list to {filename}...")
    with open(filename, 'w') as f:
        for src, dst in edges:
            f.write(f"{src},{dst}\n")
    print(f"Saved successfully!")


def main():
    print("=" * 60)
    print("Graph500 Kronecker Graph Generator for Windows")
    print("=" * 60)
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("\nUsage: python kronecker_generator.py <SCALE> [edgefactor] [format]")
        print("\nExamples:")
        print("  python kronecker_generator.py 13           # 8,192 x 8,192")
        print("  python kronecker_generator.py 14           # 16,384 x 16,384")
        print("  python kronecker_generator.py 13 16 dense  # Dense matrix")
        print("  python kronecker_generator.py 13 16 edges  # Edge list only")
        print("\nRecommended SCALE values:")
        print("  12 -> 4,096 x 4,096")
        print("  13 -> 8,192 x 8,192")
        print("  14 -> 16,384 x 16,384")
        print("  15 -> 32,768 x 32,768 (requires ~8GB RAM)")
        print("  16 -> 65,536 x 65,536 (requires ~32GB RAM)")
        sys.exit(1)
    
    scale = int(sys.argv[1])
    edgefactor = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    format_type = sys.argv[3].lower() if len(sys.argv) > 3 else 'dense'
    
    if scale > 16:
        print(f"\nWARNING: SCALE={scale} will create a {2**scale}x{2**scale} matrix")
        print(f"This requires approximately {(2**scale)**2 * 8 / 1e9:.1f} GB of RAM")
        response = input("Continue? (yes/no): ")
        if response.lower() != 'yes':
            sys.exit(0)
    
    start_time = time.time()
    
    # Generate edges
    edges, n_vertices = generate_kronecker_edges(scale, edgefactor, seed=42)
    
    edge_filename = f"kronecker_s{scale}_edges.txt"
    save_edge_list(edges, edge_filename)
    
    if format_type == 'dense':
        # Convert to dense matrix
        A = edges_to_adjacency_matrix(edges, n_vertices, symmetric=True)
        
        matrix_filename = f"kronecker_s{scale}_matrix.csv"
        save_matrix_csv(A, matrix_filename)
        
        # Print statistics
        print(f"\n" + "=" * 60)
        print("Matrix Statistics:")
        print(f"  Size: {n_vertices} x {n_vertices}")
        print(f"  Non-zeros: {np.count_nonzero(A):,}")
        print(f"  Density: {np.count_nonzero(A) / (n_vertices**2) * 100:.4f}%")
        print(f"  Memory: {A.nbytes / 1e6:.1f} MB")
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.2f} seconds")
    print("=" * 60)
    
    print(f"\nFiles generated:")
    print(f"  - {edge_filename}")
    if format_type == 'dense':
        print(f"  - {matrix_filename}")
        print(f"\nTo use with bcv_svd.c:")
        print(f"  ./bcv_svd {matrix_filename} {n_vertices} {n_vertices} 16 5")


if __name__ == "__main__":
    main()