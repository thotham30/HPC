/* bcv_nosvd_mp.c
 * 
 * Hybrid CUDA + OpenMP parallelization for BCV-Jacobi algorithm.
 * 
 * Key insight: The (q,p) loop has DATA DEPENDENCIES and CANNOT be parallelized.
 * We can only parallelize:
 * 1. Operations within Givens rotations (if m is large enough) - CUDA for large, OpenMP for medium
 * 2. Column normalization - CUDA for all columns in parallel
 * 
 * Architecture:
 * - CUDA handles large matrix operations (m > 1024)
 * - OpenMP handles coordination, medium-sized operations (512 < m <= 1024), and small operations (m <= 512)
 * 
 * Compile:
 *   nvcc -O3 -arch=sm_75 -Xcompiler -fopenmp bcv_nosvd_mp.c -o bcv_nosvd_mp -lcublas -lm
 *   (Adjust -arch based on GPU compute capability)
 */

#define _POSIX_C_SOURCE 200112L
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

double wall_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

#define A_AT(A,m,row,col) ((A)[(size_t)(col) * (m) + (row)])

/* CUDA error checking macros */
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, err); \
        exit(1); \
    } \
} while(0)

/* Thresholds for choosing execution path */
#define CUDA_THRESHOLD 1024
#define OPENMP_THRESHOLD 512

/* Global CUDA state */
static int cuda_initialized = 0;
static cudaDeviceProp device_prop;

static double *aligned_alloc_d(size_t elems) {
    void *ptr = NULL;
    size_t bytes = elems * sizeof(double);
    if (posix_memalign(&ptr, 64, bytes) != 0) return NULL;
    memset(ptr, 0, bytes);
    return (double*)ptr;
}

/* CUDA initialization */
static int init_cuda(void) {
    if (cuda_initialized) return 0;
    
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return -1;
    }
    
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, 0));
    
    printf("CUDA initialized: Device %s (Compute %d.%d, %d SMs)\n",
           device_prop.name, device_prop.major, device_prop.minor, device_prop.multiProcessorCount);
    
    cuda_initialized = 1;
    return 0;
}

/* CUDA cleanup */
static void cleanup_cuda(void) {
    if (cuda_initialized) {
        CUDA_CHECK(cudaDeviceReset());
        cuda_initialized = 0;
    }
}

/* CUDA memory allocation wrapper */
static double *cuda_malloc_d(size_t elems) {
    double *ptr = NULL;
    size_t bytes = elems * sizeof(double);
    CUDA_CHECK(cudaMalloc((void**)&ptr, bytes));
    return ptr;
}

/* CUDA memory free wrapper */
static void cuda_free_d(double *ptr) {
    if (ptr) CUDA_CHECK(cudaFree(ptr));
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

/* CUDA kernel: Compute dot products for Givens rotation (alpha, beta, gamma) */
__global__ void givens_dot_product_kernel(const double *pi, const double *pj, int m,
                                           double *alpha, double *beta, double *gamma) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double local_alpha = 0.0, local_beta = 0.0, local_gamma = 0.0;
    
    for (int r = idx; r < m; r += gridDim.x * blockDim.x) {
        double ui = pi[r], uj = pj[r];
        local_alpha += ui * ui;
        local_beta  += uj * uj;
        local_gamma += ui * uj;
    }
    
    // Reduction within block - use dynamic shared memory size
    extern __shared__ double s_data[];
    double *s_alpha = s_data;
    double *s_beta = s_data + blockDim.x;
    double *s_gamma = s_data + 2 * blockDim.x;
    
    int tid = threadIdx.x;
    if (tid < blockDim.x) {
        s_alpha[tid] = local_alpha;
        s_beta[tid] = local_beta;
        s_gamma[tid] = local_gamma;
    }
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockDim.x) {
            s_alpha[tid] += s_alpha[tid + s];
            s_beta[tid] += s_beta[tid + s];
            s_gamma[tid] += s_gamma[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(alpha, s_alpha[0]);
        atomicAdd(beta, s_beta[0]);
        atomicAdd(gamma, s_gamma[0]);
    }
}

/* CUDA kernel: Apply Givens rotation to matrix columns */
__global__ void givens_apply_rotation_kernel(double *pi, double *pj, int m, double c, double s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        double ui = pi[idx], uj = pj[idx];
        pi[idx] = c * ui - s * uj;
        pj[idx] = s * ui + c * uj;
    }
}

/* CUDA kernel: Normalize a single column */
__global__ void normalize_column_kernel(double *colptr, int m, double *norm_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double local_sum = 0.0;
    
    for (int i = idx; i < m; i += gridDim.x * blockDim.x) {
        local_sum += colptr[i] * colptr[i];
    }
    
    // Reduction within block - use dynamic shared memory
    extern __shared__ double s_sum[];
    int tid = threadIdx.x;
    if (tid < blockDim.x) {
        s_sum[tid] = local_sum;
    }
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockDim.x) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(norm_out, s_sum[0]);
    }
}

/* CUDA kernel: Apply normalization factor to column */
__global__ void apply_normalization_kernel(double *colptr, int m, double inv_norm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        colptr[idx] *= inv_norm;
    }
}

/* Givens rotation with hybrid CUDA/OpenMP/Serial execution */
void givens_rotation_2k(double *U, int m, int k) {
    int two_k = 2 * k;
    int use_cuda = (m > CUDA_THRESHOLD) && cuda_initialized;
    int use_openmp = (m > OPENMP_THRESHOLD && m <= CUDA_THRESHOLD) && (omp_get_max_threads() > 1);
    
    if (use_cuda) {
        // CUDA path: Transfer to GPU, compute, transfer back
        double *d_U = NULL;
        double *d_alpha = NULL, *d_beta = NULL, *d_gamma = NULL;
        size_t U_size = (size_t)m * two_k * sizeof(double);
        
        CUDA_CHECK(cudaMalloc((void**)&d_U, U_size));
        CUDA_CHECK(cudaMalloc((void**)&d_alpha, sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&d_beta, sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&d_gamma, sizeof(double)));
        
        CUDA_CHECK(cudaMemcpy(d_U, U, U_size, cudaMemcpyHostToDevice));
        
        int threads_per_block = 256;
        int num_blocks = (m + threads_per_block - 1) / threads_per_block;
        
        for (int i = 0; i < two_k - 1; ++i) {
            for (int j = i + 1; j < two_k; ++j) {
                double *d_pi = d_U + (size_t)i * m;
                double *d_pj = d_U + (size_t)j * m;
                
                // Reset reduction variables
                CUDA_CHECK(cudaMemset(d_alpha, 0, sizeof(double)));
                CUDA_CHECK(cudaMemset(d_beta, 0, sizeof(double)));
                CUDA_CHECK(cudaMemset(d_gamma, 0, sizeof(double)));
                
                // Compute dot products on GPU
                size_t shared_mem_size = 3 * threads_per_block * sizeof(double);
                givens_dot_product_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
                    d_pi, d_pj, m, d_alpha, d_beta, d_gamma);
                CUDA_CHECK(cudaDeviceSynchronize());
                
                // Copy results back
                double alpha, beta, gamma;
                CUDA_CHECK(cudaMemcpy(&alpha, d_alpha, sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&beta, d_beta, sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&gamma, d_gamma, sizeof(double), cudaMemcpyDeviceToHost));
                
                if (fabs(gamma) < 1e-14) continue;
                
                // Compute rotation parameters on CPU
                double tau = (beta - alpha) / (2.0 * gamma);
                double t = (tau >= 0.0 ? 1.0 : -1.0) / (fabs(tau) + sqrt(1.0 + tau * tau));
                double c = 1.0 / sqrt(1.0 + t * t);
                double s = t * c;
                
                // Apply rotation on GPU
                givens_apply_rotation_kernel<<<num_blocks, threads_per_block>>>(
                    d_pi, d_pj, m, c, s);
                CUDA_CHECK(cudaDeviceSynchronize());
            }
        }
        
        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(U, d_U, U_size, cudaMemcpyDeviceToHost));
        
        cuda_free_d(d_U);
        cuda_free_d(d_alpha);
        cuda_free_d(d_beta);
        cuda_free_d(d_gamma);
    } else if (use_openmp) {
        // OpenMP path: Medium-sized matrices
        for (int i = 0; i < two_k - 1; ++i) {
            for (int j = i + 1; j < two_k; ++j) {
                double *pi = U + (size_t)i * m;
                double *pj = U + (size_t)j * m;
                
                double alpha = 0.0, beta = 0.0, gamma = 0.0;
                
                #pragma omp parallel for reduction(+:alpha,beta,gamma) schedule(static,1024)
                for (int r = 0; r < m; ++r) {
                    double ui = pi[r], uj = pj[r];
                    alpha += ui * ui;
                    beta  += uj * uj;
                    gamma += ui * uj;
                }

                if (fabs(gamma) < 1e-14) continue;
                
                double tau = (beta - alpha) / (2.0 * gamma);
                double t = (tau >= 0.0 ? 1.0 : -1.0) / (fabs(tau) + sqrt(1.0 + tau * tau));
                double c = 1.0 / sqrt(1.0 + t * t);
                double s = t * c;

                #pragma omp parallel for schedule(static,1024)
                for (int r = 0; r < m; ++r) {
                    double ui = pi[r], uj = pj[r];
                    pi[r] = c * ui - s * uj;
                    pj[r] = s * ui + c * uj;
                }
            }
        }
    } else {
        // Serial path: Small matrices
        for (int i = 0; i < two_k - 1; ++i) {
            for (int j = i + 1; j < two_k; ++j) {
                double *pi = U + (size_t)i * m;
                double *pj = U + (size_t)j * m;
                
                double alpha = 0.0, beta = 0.0, gamma = 0.0;
                
                for (int r = 0; r < m; ++r) {
                    double ui = pi[r], uj = pj[r];
                    alpha += ui * ui;
                    beta  += uj * uj;
                    gamma += ui * uj;
                }

                if (fabs(gamma) < 1e-14) continue;
                
                double tau = (beta - alpha) / (2.0 * gamma);
                double t = (tau >= 0.0 ? 1.0 : -1.0) / (fabs(tau) + sqrt(1.0 + tau * tau));
                double c = 1.0 / sqrt(1.0 + t * t);
                double s = t * c;

                for (int r = 0; r < m; ++r) {
                    double ui = pi[r], uj = pj[r];
                    pi[r] = c * ui - s * uj;
                    pj[r] = s * ui + c * uj;
                }
            }
        }
    }
}

/* Normalize columns with hybrid CUDA/OpenMP execution */
static void normalize_columns_hybrid(double *A, int m, int n) {
    int use_cuda = (m > CUDA_THRESHOLD) && cuda_initialized;
    
    if (use_cuda) {
        // CUDA path: Normalize all columns on GPU
        double *d_A = NULL;
        double *d_norms = NULL;
        size_t A_size = (size_t)m * n * sizeof(double);
        
        CUDA_CHECK(cudaMalloc((void**)&d_A, A_size));
        CUDA_CHECK(cudaMalloc((void**)&d_norms, (size_t)n * sizeof(double)));
        
        CUDA_CHECK(cudaMemcpy(d_A, A, A_size, cudaMemcpyHostToDevice));
        
        int threads_per_block = 256;
        int num_blocks = (m + threads_per_block - 1) / threads_per_block;
        
        // OpenMP manages parallel column processing
        #pragma omp parallel for schedule(static, 8)
        for (int col = 0; col < n; ++col) {
            double *d_colptr = d_A + (size_t)col * m;
            double *d_norm = d_norms + col;
            
            CUDA_CHECK(cudaMemset(d_norm, 0, sizeof(double)));
            
            // Compute norm
            size_t shared_mem_size = threads_per_block * sizeof(double);
            normalize_column_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
                d_colptr, m, d_norm);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Get norm
            double nrm;
            CUDA_CHECK(cudaMemcpy(&nrm, d_norm, sizeof(double), cudaMemcpyDeviceToHost));
            nrm = sqrt(nrm);
            
            if (nrm > 1e-14) {
                double inv = 1.0 / nrm;
                apply_normalization_kernel<<<num_blocks, threads_per_block>>>(
                    d_colptr, m, inv);
                CUDA_CHECK(cudaDeviceSynchronize());
            }
        }
        
        // Copy result back
        CUDA_CHECK(cudaMemcpy(A, d_A, A_size, cudaMemcpyDeviceToHost));
        
        cuda_free_d(d_A);
        cuda_free_d(d_norms);
    } else {
        // OpenMP path: CPU parallel normalization
        #pragma omp parallel for schedule(static, 16)
        for (int col = 0; col < n; ++col) {
            double s = 0.0;
            double *colptr = A + (size_t)col * m;
            
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

    // Initialize CUDA
    if (init_cuda() != 0) {
        fprintf(stderr, "CUDA initialization failed - CUDA is required\n");
        return 1;
    }

    int nthreads = omp_get_max_threads();
    printf("BCV-Jacobi (CUDA+OpenMP hybrid) on %s (%dx%d) k=%d sweeps=%d threads=%d\n",
           csvname, m, n, k, sweeps, nthreads);

    double *A = aligned_alloc_d((size_t)m * n);
    if (!A || load_csv_submatrix(csvname, A, m, n) != 0) {
        fprintf(stderr, "Failed to load matrix\n");
        cleanup_cuda();
        return 1;
    }
    printf("Matrix loaded successfully.\n");

    int two_k = 2 * k;
    double *U = aligned_alloc_d((size_t)m * two_k);
    if (!U) { free(A); cleanup_cuda(); return 1; }

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
        
        // Normalize columns - Hybrid CUDA/OpenMP
        normalize_columns_hybrid(A, m, n);
    }
    
    double t1 = wall_time();
    printf("Elapsed time = %.6f s\n", t1 - t0);

    if (save_matrix_csv(outA, A, m, n) == 0) {
        printf("Saved output to %s\n", outA);
    }

    free(A); free(U);
    cleanup_cuda();
    return 0;
}