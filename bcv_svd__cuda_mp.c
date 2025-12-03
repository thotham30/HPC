/* bcv_svd_mp.c
 *
 * Hybrid CUDA + OpenMP BCV-Jacobi implementation with SVD.
 *
 * Architecture:
 * - CUDA (cuBLAS) handles large matrix multiplications (GEMM)
 * - OpenMP handles coordination, small matrix operations (jacobi_eigen), and memory management
 * - Both technologies work together for optimal performance
 *
 * Compile:
 *   nvcc -O3 -arch=sm_75 -Xcompiler -fopenmp bcv_svd_mp.c -o bcv_svd_mp -lcublas -lm
 *   (Adjust -arch based on GPU compute capability)
 *
 * Run:
 *   OMP_NUM_THREADS=4 ./bcv_svd_mp <csv> <m> <n> <k> <sweeps> <outA> <outV>
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200112L
#endif

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <sys/types.h>
#include <string.h>
#include <errno.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

double wall_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

#define A_AT(A,m,row,col) ((A)[ (size_t)(col) * (m) + (row) ])

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
static cublasHandle_t cublas_handle = NULL;
static cudaDeviceProp device_prop;

static double *aligned_alloc_d(size_t elems) {
    if (elems == 0) return NULL;
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
    
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    
    printf("CUDA initialized: Device %s (Compute %d.%d, %d SMs)\n",
           device_prop.name, device_prop.major, device_prop.minor, device_prop.multiProcessorCount);
    
    cuda_initialized = 1;
    return 0;
}

/* CUDA cleanup */
static void cleanup_cuda(void) {
    if (cublas_handle) {
        cublasDestroy(cublas_handle);
        cublas_handle = NULL;
    }
    if (cuda_initialized) {
        CUDA_CHECK(cudaDeviceReset());
        cuda_initialized = 0;
    }
}

/* CUDA memory allocation wrapper */
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
            if (j == cols - 1)
                fprintf(fp, "%.15g", val);
            else
                fprintf(fp, "%.15g,", val);
        }
        fputc('\n', fp);
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
    int row = 0;
    double *rowvals = (double*)malloc(sizeof(double) * n);
    if (!rowvals) { fclose(fp); return -2; }

    while ((getline(&line, &len, fp)) != -1 && row < m) {
        int col = 0;
        char *ptr = line, *endptr;
        while (col < n) {
            while (*ptr == ' ' || *ptr == '\t') ++ptr;
            if (*ptr == '\0' || *ptr == '\n' || *ptr == '\r') break;
            double v = strtod(ptr, &endptr);
            if (ptr == endptr) { if (*ptr == ',') { ++ptr; continue; } break; }
            rowvals[col] = v;
            ++col;
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

static void init_V_identity(double *V, int n) {
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            V[(size_t)j * n + i] = (i == j) ? 1.0 : 0.0;
}

/* CUDA kernel: Normalize a single column */
__global__ void normalize_column_kernel_svd(double *colptr, int m, double *norm_out) {
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
__global__ void apply_normalization_kernel_svd(double *colptr, int m, double inv_norm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        colptr[idx] *= inv_norm;
    }
}

/* Hybrid dgemm: cuBLAS for large matrices, OpenMP for smaller ones */
static void dgemm_hybrid(char opA, char opB,
                         int m, int n, int k,
                         double alpha,
                         const double *A, int lda,
                         const double *B, int ldb,
                         double beta,
                         double *C, int ldc)
{
    // Use cuBLAS for large matrices, OpenMP for smaller ones
    int use_cublas = (m > CUDA_THRESHOLD || n > CUDA_THRESHOLD || k > CUDA_THRESHOLD) && cuda_initialized;
    
    if (use_cublas) {
        // cuBLAS path: Transfer to GPU, compute, transfer back
        // Note: Our matrices use column-major storage (A_AT macro), so cuBLAS works directly
        double *d_A = NULL, *d_B = NULL, *d_C = NULL;
        int A_cols = (opA == 'N') ? k : m;
        int B_cols = (opB == 'N') ? n : k;
        size_t A_size = (size_t)lda * A_cols * sizeof(double);
        size_t B_size = (size_t)ldb * B_cols * sizeof(double);
        size_t C_size = (size_t)ldc * n * sizeof(double);
        
        CUDA_CHECK(cudaMalloc((void**)&d_A, A_size));
        CUDA_CHECK(cudaMalloc((void**)&d_B, B_size));
        CUDA_CHECK(cudaMalloc((void**)&d_C, C_size));
        
        CUDA_CHECK(cudaMemcpy(d_A, A, A_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B, B_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_C, C, C_size, cudaMemcpyHostToDevice));
        
        cublasOperation_t cublas_opA = (opA == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
        cublasOperation_t cublas_opB = (opB == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
        
        // cuBLAS: C = alpha * op(A) * op(B) + beta * C (column-major)
        // Function computes: C = opA(A) * opB(B) where C is m×n
        // For opA='T': A^T is k×m, for opA='N': A is m×k
        // For opB='T': B^T is n×k, for opB='N': B is k×n
        // Common dimension between A and B is k
        int common_dim = k;
        
        CUBLAS_CHECK(cublasDgemm(cublas_handle, cublas_opA, cublas_opB,
                                  m, n, common_dim,
                                  &alpha, d_A, lda, d_B, ldb,
                                  &beta, d_C, ldc));
        
        CUDA_CHECK(cudaMemcpy(C, d_C, C_size, cudaMemcpyDeviceToHost));
        
        cuda_free_d(d_A);
        cuda_free_d(d_B);
        cuda_free_d(d_C);
    } else {
        // OpenMP path: CPU parallel matrix multiplication
        #pragma omp for schedule(static)
        for (int jc = 0; jc < n; ++jc) {
            for (int ic = 0; ic < m; ++ic) {
                double sum = 0.0;
                if (opA == 'N' && opB == 'N') {
                    for (int l = 0; l < k; ++l)
                        sum += A[(size_t)l * lda + ic] * B[(size_t)jc * ldb + l];
                } else if (opA == 'T' && opB == 'N') {
                    for (int l = 0; l < k; ++l)
                        sum += A[(size_t)ic * lda + l] * B[(size_t)jc * ldb + l];
                } else if (opA == 'N' && opB == 'T') {
                    for (int l = 0; l < k; ++l)
                        sum += A[(size_t)l * lda + ic] * B[(size_t)l * ldb + jc];
                } else {
                    for (int l = 0; l < k; ++l)
                        sum += A[(size_t)ic * lda + l] * B[(size_t)l * ldb + jc];
                }
                double cval = C[(size_t)jc * ldc + ic];
                C[(size_t)jc * ldc + ic] = alpha * sum + beta * cval;
            }
        }
    }
}

static void jacobi_eigen_small(double *G, double *R, int k, int max_iter, double tol) {
    for (int j = 0; j < k; ++j)
        for (int i = 0; i < k; ++i)
            R[(size_t)j * k + i] = (i == j) ? 1.0 : 0.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        double max_off = 0.0; int p = -1, q = -1;
        for (int col = 0; col < k; ++col)
            for (int row = 0; row < col; ++row) {
                double a = fabs(G[(size_t)col * k + row]);
                if (a > max_off) { max_off = a; p = row; q = col; }
            }
        if (max_off < tol) break;
        double App = G[(size_t)p * k + p];
        double Aqq = G[(size_t)q * k + q];
        double Apq = G[(size_t)q * k + p];
        if (fabs(Apq) < 1e-18) continue;
        double tau = (Aqq - App) / (2.0 * Apq);
        double t = (tau >= 0.0 ? 1.0 : -1.0) / (fabs(tau) + sqrt(1.0 + tau*tau));
        double c = 1.0 / sqrt(1.0 + t*t);
        double s = t * c;
        for (int r = 0; r < k; ++r) {
            if (r == p || r == q) continue;
            double Grp = G[(size_t)p * k + r];
            double Grq = G[(size_t)q * k + r];
            double new_rp = c * Grp - s * Grq;
            double new_rq = s * Grp + c * Grq;
            G[(size_t)p * k + r] = new_rp;
            G[(size_t)r * k + p] = new_rp;
            G[(size_t)q * k + r] = new_rq;
            G[(size_t)r * k + q] = new_rq;
        }
        double new_pp = c*c*App - 2.0*s*c*Apq + s*s*Aqq;
        double new_qq = s*s*App + 2.0*s*c*Apq + c*c*Aqq;
        G[(size_t)p * k + p] = new_pp;
        G[(size_t)q * k + q] = new_qq;
        G[(size_t)q * k + p] = 0.0;
        G[(size_t)p * k + q] = 0.0;
        for (int r = 0; r < k; ++r) {
            double Rip = R[(size_t)p * k + r];
            double Riq = R[(size_t)q * k + r];
            R[(size_t)p * k + r] = c * Rip - s * Riq;
            R[(size_t)q * k + r] = s * Rip + c * Riq;
        }
    }
}

static void normalize_columns(double *A, int m, int n) {
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
        #pragma omp for schedule(static, 8)
        for (int col = 0; col < n; ++col) {
            double *d_colptr = d_A + (size_t)col * m;
            double *d_norm = d_norms + col;
            
            CUDA_CHECK(cudaMemset(d_norm, 0, sizeof(double)));
            
            // Compute norm
            size_t shared_mem_size = threads_per_block * sizeof(double);
            normalize_column_kernel_svd<<<num_blocks, threads_per_block, shared_mem_size>>>(
                d_colptr, m, d_norm);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Get norm
            double nrm;
            CUDA_CHECK(cudaMemcpy(&nrm, d_norm, sizeof(double), cudaMemcpyDeviceToHost));
            nrm = sqrt(nrm);
            
            if (nrm > 1e-14) {
                double inv = 1.0 / nrm;
                apply_normalization_kernel_svd<<<num_blocks, threads_per_block>>>(
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
        #pragma omp for schedule(static, 8)
        for (int col = 0; col < n; ++col) {
            double s = 0.0;
            double *colptr = A + (size_t)col * m;
            for (int i = 0; i < m; ++i) s += colptr[i] * colptr[i];
            double nrm = sqrt(s);
            if (nrm > 1e-14) {
                double inv = 1.0 / nrm;
                for (int i = 0; i < m; ++i) colptr[i] *= inv;
            }
        }
    }
}

int main(int argc, char **argv) {
    const char *csvname = (argc > 1) ? argv[1] : "orl_matrix.csv";
    int m = (argc > 2) ? atoi(argv[2]) : 128;
    int n = (argc > 3) ? atoi(argv[3]) : 128;
    int k = (argc > 4) ? atoi(argv[4]) : 16;
    int sweeps = (argc > 5) ? atoi(argv[5]) : 5;
    const char *outA = (argc > 6) ? argv[6] : "output_A.csv";
    const char *outV = (argc > 7) ? argv[7] : NULL;

    if (n % k != 0) { fprintf(stderr, "n must be divisible by k\n"); return 1; }

    // Initialize CUDA
    if (init_cuda() != 0) {
        fprintf(stderr, "CUDA initialization failed - CUDA is required\n");
        return 1;
    }

    int nthreads = omp_get_max_threads();
    printf("BCV-Jacobi WITH V (CUDA+OpenMP hybrid) on %s (%dx%d), k=%d, sweeps=%d, threads=%d\n",
           csvname, m, n, k, sweeps, nthreads);

    size_t m_n = (size_t)m * n, n_n = (size_t)n * n, two_k = (size_t)2 * k;
    double *A = aligned_alloc_d(m_n);
    double *V = aligned_alloc_d(n_n);
    double *Ubuf = aligned_alloc_d((size_t)m * two_k);
    double *G = aligned_alloc_d(two_k * two_k);
    double *R = aligned_alloc_d(two_k * two_k);
    double *Utmp = aligned_alloc_d((size_t)m * two_k);
    double *Vsub = aligned_alloc_d((size_t)n * two_k);
    double *Vtmp = aligned_alloc_d((size_t)n * two_k);

    if (!A || !V || !Ubuf || !G || !R || !Utmp || !Vsub || !Vtmp) {
        fprintf(stderr,"Memory allocation failed (errno=%d)\n", errno);
        free(A); free(V); free(Ubuf); free(G); free(R); free(Utmp); free(Vsub); free(Vtmp);
        return 1;
    }

    if (load_csv_submatrix(csvname, A, m, n) != 0) {
        fprintf(stderr, "Failed to load %s as %dx%d matrix\n", csvname, m, n);
        free(A); free(V); free(Ubuf); free(G); free(R); free(Utmp); free(Vsub); free(Vtmp);
        return 1;
    }
    printf("Matrix loaded successfully.\n");

    init_V_identity(V, n);
    int blocks = n / k;
    double t0 = wall_time();

    #pragma omp parallel
    {
        for (int sweep = 0; sweep < sweeps; ++sweep) {
            for (int q = 0; q < blocks - 1; ++q) {
                /* Master loads q-block into Ubuf[0:k-1] */
                #pragma omp master
                {
                    for (int jj = 0; jj < k; ++jj)
                        memcpy(Ubuf + (size_t)jj * m, A + (size_t)(q*k + jj) * m, sizeof(double) * m);
                }
                #pragma omp barrier

                for (int p = q + 1; p < blocks; ++p) {
                    /* Master loads p-block into Ubuf[k:2k-1] */
                    #pragma omp master
                    {
                        for (int jj = 0; jj < k; ++jj)
                            memcpy(Ubuf + (size_t)(k + jj) * m, A + (size_t)(p*k + jj) * m, sizeof(double) * m);
                    }
                    #pragma omp barrier

                    int tk = 2 * k;

                    /* G = Ubuf^T * Ubuf - Hybrid cuBLAS/OpenMP */
                    dgemm_hybrid('T', 'N', tk, tk, m, 1.0, Ubuf, m, Ubuf, m, 0.0, G, tk);
                    #pragma omp barrier

                    /* Jacobi eigendecomposition (serial on CPU - too small for GPU) */
                    #pragma omp master
                    {
                        jacobi_eigen_small(G, R, tk, 200, 1e-12);
                    }
                    #pragma omp barrier

                    /* Utmp = Ubuf * R - Hybrid cuBLAS/OpenMP */
                    dgemm_hybrid('N', 'N', m, tk, tk, 1.0, Ubuf, m, R, tk, 0.0, Utmp, m);
                    #pragma omp barrier

                    /* Master updates Ubuf from Utmp */
                    #pragma omp master
                    {
                        memcpy(Ubuf, Utmp, sizeof(double) * (size_t)m * tk);
                        
                        /* Write transformed p-block back to A immediately */
                        for (int jj = 0; jj < k; ++jj)
                            memcpy(A + (size_t)(p * k + jj) * m, Ubuf + (size_t)(k + jj) * m, sizeof(double) * m);
                        
                        /* Gather V blocks for transformation */
                        for (int jj = 0; jj < k; ++jj)
                            memcpy(Vsub + (size_t)jj * n, V + (size_t)(q * k + jj) * n, sizeof(double) * n);
                        for (int jj = 0; jj < k; ++jj)
                            memcpy(Vsub + (size_t)(k + jj) * n, V + (size_t)(p * k + jj) * n, sizeof(double) * n);
                    }
                    #pragma omp barrier

                    /* Vtmp = Vsub * R - Hybrid cuBLAS/OpenMP */
                    dgemm_hybrid('N', 'N', n, tk, tk, 1.0, Vsub, n, R, tk, 0.0, Vtmp, n);
                    #pragma omp barrier

                    /* Master scatters transformed V back */
                    #pragma omp master
                    {
                        for (int jj = 0; jj < k; ++jj)
                            memcpy(V + (size_t)(q * k + jj) * n, Vtmp + (size_t)jj * n, sizeof(double) * n);
                        for (int jj = 0; jj < k; ++jj)
                            memcpy(V + (size_t)(p * k + jj) * n, Vtmp + (size_t)(k + jj) * n, sizeof(double) * n);
                    }
                    #pragma omp barrier
                }

                /* After all p iterations, write back q-block */
                #pragma omp master
                {
                    for (int jj = 0; jj < k; ++jj)
                        memcpy(A + (size_t)(q * k + jj) * m, Ubuf + (size_t)jj * m, sizeof(double) * m);
                }
                #pragma omp barrier
            }

            /* Normalize columns */
            normalize_columns(A, m, n);
            #pragma omp barrier
        }
    }

    double t1 = wall_time();
    printf("Elapsed time (BCV with V, CUDA+OpenMP hybrid) = %.6f seconds\n", t1 - t0);

    if (outA && strlen(outA) > 0) {
        printf("Saving output matrix A to '%s' ...\n", outA);
        if (save_matrix_csv(outA, A, m, n) != 0) {
            fprintf(stderr, "Failed to save A to %s\n", outA);
        } else {
            printf("Saved A to %s\n", outA);
        }
    }
    if (outV && strlen(outV) > 0) {
        printf("Saving V matrix to '%s' ...\n", outV);
        if (save_matrix_csv(outV, V, n, n) != 0) {
            fprintf(stderr, "Failed to save V to %s\n", outV);
        } else {
            printf("Saved V to %s\n", outV);
        }
    }

    free(A); free(V); free(Ubuf); free(G); free(R); free(Utmp); free(Vsub); free(Vtmp);
    cleanup_cuda();
    return 0;
}