/* bcv_nosvd_mp_CORRECT.c
 * 
 * CORRECT parallelization that produces SAME results as serial version.
 * 
 * Key insight: The (q,p) loop has DATA DEPENDENCIES and CANNOT be parallelized.
 * We can only parallelize:
 * 1. Operations within Givens rotations (if m is large enough)
 * 2. Column normalization
 * 
 * This version focuses on correctness first, performance second.
 * 
 * Compile:
 *   gcc -O3 -march=native -fopenmp bcv_nosvd_mp_CORRECT.c -o bcv_nosvd_mp_correct -lm
 */

#define _POSIX_C_SOURCE 200112L
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <omp.h>

double wall_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

#define A_AT(A,m,row,col) ((A)[(size_t)(col) * (m) + (row)])

static double *aligned_alloc_d(size_t elems) {
    void *ptr = NULL;
    size_t bytes = elems * sizeof(double);
    if (posix_memalign(&ptr, 64, bytes) != 0) return NULL;
    memset(ptr, 0, bytes);
    return (double*)ptr;
}

static int save_matrix_csv(const char *fname, double *M, int rows, int cols) {
    FILE *fp = fopen(fname, "w");
    if (!fp) return -1;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double val = A_AT(M, rows, i, j);
            fprintf(fp, (j == cols - 1) ? "%.15g\n" : "%.15g,", val);
        }
    }
    fclose(fp);
    return 0;
}

static int load_csv_submatrix(const char *fname, double *A, int m, int n) {
    FILE *fp = fopen(fname, "r");
    if (!fp) {
        fprintf(stderr, "Failed to open CSV file: %s\n", fname);
        return -1;
    }
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    int row = 0;
    double *rowvals = (double*)malloc(sizeof(double) * n);
    if (!rowvals) { fclose(fp); return -2; }

    while ((read = getline(&line, &len, fp)) != -1 && row < m) {
        int col = 0;
        char *ptr = line, *endptr;
        while (col < n) {
            while (*ptr == ' ' || *ptr == '\t') ++ptr;
            if (*ptr == '\0' || *ptr == '\n' || *ptr == '\r') break;
            double v = strtod(ptr, &endptr);
            if (ptr == endptr) { if (*ptr == ',') { ++ptr; continue; } break; }
            rowvals[col++] = v;
            ptr = endptr;
            while (*ptr == ' ' || *ptr == '\t') ++ptr;
            if (*ptr == ',') ++ptr;
        }
        if (col < n) {
            fprintf(stderr, "CSV row %d has only %d cols (need %d)\n", row+1, col, n);
            free(rowvals); if (line) free(line); fclose(fp); return -3;
        }
        for (int j = 0; j < n; ++j)
            A_AT(A, m, row, j) = rowvals[j];
        ++row;
    }

    if (line) free(line);
    free(rowvals);
    fclose(fp);
    if (row < m) {
        fprintf(stderr, "CSV has only %d rows (need %d)\n", row, m);
        return -4;
    }
    return 0;
}

void load_block(double *dst, const double *A, int m, int start_col, int k) {
    for (int j = 0; j < k; ++j) {
        memcpy(dst + (size_t)j * m, 
               A + (size_t)(start_col + j) * m, 
               m * sizeof(double));
    }
}

void store_block(double *A, const double *src, int m, int start_col, int k) {
    for (int j = 0; j < k; ++j) {
        memcpy(A + (size_t)(start_col + j) * m,
               src + (size_t)j * m,
               m * sizeof(double));
    }
}

/* Givens rotation with OPTIONAL parallelization only for VERY large m */
void givens_rotation_2k(double *U, int m, int k) {
    int two_k = 2 * k;
    
    // Only parallelize if m is extremely large AND we have threads available
    // Otherwise serial is faster due to thread overhead
    const int PARALLEL_THRESHOLD = 32768;  // Very high threshold
    int use_parallel = (m >= PARALLEL_THRESHOLD) && (omp_get_max_threads() > 1);
    
    for (int i = 0; i < two_k - 1; ++i) {
        for (int j = i + 1; j < two_k; ++j) {
            double *pi = U + (size_t)i * m;
            double *pj = U + (size_t)j * m;
            
            double alpha = 0.0, beta = 0.0, gamma = 0.0;
            
            if (use_parallel) {
                #pragma omp parallel for reduction(+:alpha,beta,gamma) schedule(static,1024)
                for (int r = 0; r < m; ++r) {
                    double ui = pi[r], uj = pj[r];
                    alpha += ui * ui;
                    beta  += uj * uj;
                    gamma += ui * uj;
                }
            } else {
                for (int r = 0; r < m; ++r) {
                    double ui = pi[r], uj = pj[r];
                    alpha += ui * ui;
                    beta  += uj * uj;
                    gamma += ui * uj;
                }
            }

            if (fabs(gamma) < 1e-14) continue;
            
            double tau = (beta - alpha) / (2.0 * gamma);
            double t = (tau >= 0.0 ? 1.0 : -1.0) / (fabs(tau) + sqrt(1.0 + tau * tau));
            double c = 1.0 / sqrt(1.0 + t * t);
            double s = t * c;

            if (use_parallel) {
                #pragma omp parallel for schedule(static,1024)
                for (int r = 0; r < m; ++r) {
                    double ui = pi[r], uj = pj[r];
                    pi[r] = c * ui - s * uj;
                    pj[r] = s * ui + c * uj;
                }
            } else {
                for (int r = 0; r < m; ++r) {
                    double ui = pi[r], uj = pj[r];
                    pi[r] = c * ui - s * uj;
                    pj[r] = s * ui + c * uj;
                }
            }
        }
    }
}

int main(int argc, char **argv){
    if (argc < 5) {
        fprintf(stderr, "Usage: %s input.csv m n k [sweeps] [outA.csv]\n", argv[0]);
        return 1;
    }

    const char *csvname = argv[1];
    int m = atoi(argv[2]);
    int n = atoi(argv[3]);
    int k = atoi(argv[4]);
    int sweeps = (argc > 5) ? atoi(argv[5]) : 5;
    const char *outA = (argc > 6) ? argv[6] : "output_A_mp.csv";

    if (m <= 0 || n <= 0 || k <= 0 || n % k != 0) {
        fprintf(stderr, "Invalid parameters\n");
        return 1;
    }

    int nthreads = omp_get_max_threads();
    printf("BCV-Jacobi (correct OpenMP) on %s (%dx%d) k=%d sweeps=%d threads=%d\n",
           csvname, m, n, k, sweeps, nthreads);

    double *A = aligned_alloc_d((size_t)m * n);
    if (!A || load_csv_submatrix(csvname, A, m, n) != 0) {
        fprintf(stderr, "Failed to load matrix\n");
        return 1;
    }
    printf("Matrix loaded successfully.\n");

    int two_k = 2 * k;
    double *U = aligned_alloc_d((size_t)m * two_k);
    if (!U) { free(A); return 1; }

    int blocks = n / k;
    double t0 = wall_time();
    
    for (int sweep = 0; sweep < sweeps; ++sweep) {
        // MUST be serial over q (data dependencies)
        for (int q = 0; q < blocks - 1; ++q) {
            // Load q-block into U[:,0:k] ONCE per q
            load_block(U, A, m, q * k, k);
            
            // MUST be serial over p (data dependencies - each p updates q-block)
            for (int p = q + 1; p < blocks; ++p) {
                // Load p-block into U[:,k:2k]
                load_block(U + (size_t)k * m, A, m, p * k, k);
                
                // Rotate both blocks in U
                // This updates BOTH U[:,0:k] and U[:,k:2k]
                // Can optionally parallelize INSIDE this function if m is huge
                givens_rotation_2k(U, m, k);
                
                // Store p-block (U[:,k:2k]) back to A
                store_block(A, U + (size_t)k * m, m, p * k, k);
                
                // q-block (U[:,0:k]) stays in U for next p iteration
            }
            
            // After all p iterations, store updated q-block back to A
            store_block(A, U, m, q * k, k);
        }
        
        // Normalize columns - SAFE to parallelize (independent columns)
        #pragma omp parallel for schedule(static, 16)
        for (int col = 0; col < n; ++col) {
            double s = 0.0;
            double *colptr = A + (size_t)col * m;
            
            // Could also parallelize this reduction if m is huge
            for (int i = 0; i < m; ++i) {
                s += colptr[i] * colptr[i];
            }
            
            double nrm = sqrt(s);
            if (nrm > 1e-14) {
                double inv = 1.0 / nrm;
                for (int i = 0; i < m; ++i) {
                    colptr[i] *= inv;
                }
            }
        }
    }
    
    double t1 = wall_time();
    printf("Elapsed time = %.6f s\n", t1 - t0);

    if (save_matrix_csv(outA, A, m, n) == 0) {
        printf("Saved output to %s\n", outA);
    }

    free(A); free(U);
    return 0;
}