/* bcv_svd_mp_FIXED.c
 * 
 * FIX: Use persistent thread pool instead of spawning threads repeatedly
 * 
 * Key change: Move #pragma omp parallel OUTSIDE the hot loops
 * Only use #pragma omp for inside the parallel region
 * 
 * Compile:
 *   gcc -O3 -march=native -fopenmp bcv_svd_mp_FIXED.c -o bcv_svd_mp_fixed -lm
 */

#define _POSIX_C_SOURCE 200112L
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <sys/types.h>
#include <string.h>
#include <omp.h>

double wall_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

#define A_AT(A,m,row,col) ((A)[ (size_t)(col) * (m) + (row) ])

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
    for (int j=0;j<n;++j)
        for (int i=0;i<n;++i)
            V[(size_t)j * n + i] = (i==j) ? 1.0 : 0.0;
}

/* GEMM - NO #pragma omp parallel here! 
 * Will be called from within existing parallel region
 */
static void dgemm_simple(char opA, char opB,
                         int m, int n, int k,
                         double alpha,
                         const double *A, int lda,
                         const double *B, int ldb,
                         double beta,
                         double *C, int ldc)
{
    // Use #pragma omp for (not parallel for) - assumes already in parallel region
    #pragma omp for schedule(static, 4) nowait
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

/* Jacobi - keep serial (only 2.1% of time) */
static void jacobi_eigen_small(double *G, double *R, int k, int max_iter, double tol) {
    for (int j=0;j<k;++j)
        for (int i=0;i<k;++i)
            R[(size_t)j * k + i] = (i==j) ? 1.0 : 0.0;

    for (int iter=0; iter<max_iter; ++iter) {
        double max_off = 0.0; int p=-1, q=-1;
        for (int col=0; col<k; ++col)
            for (int row=0; row<col; ++row) {
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
    // Assumes already in parallel region
    #pragma omp for schedule(static, 8)
    for (int col=0; col<n; ++col) {
        double s = 0.0;
        double *colptr = A + (size_t)col * m;
        for (int i=0;i<m;++i) s += colptr[i]*colptr[i];
        double nrm = sqrt(s);
        if (nrm > 1e-14) {
            double inv = 1.0 / nrm;
            for (int i=0;i<m;++i) colptr[i] *= inv;
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

    int nthreads = omp_get_max_threads();
    printf("BCV-Jacobi WITH V (OpenMP FIXED) on %s (%dx%d), k=%d, sweeps=%d, threads=%d\n",
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
        fprintf(stderr,"Memory allocation failed\n");
        return 1;
    }

    if (load_csv_submatrix(csvname, A, m, n) != 0) {
        fprintf(stderr, "Failed to load %s as %dx%d matrix\n", csvname, m, n);
        return 1;
    }
    printf("Matrix loaded successfully.\n");

    init_V_identity(V, n);
    int blocks = n / k;
    double t0 = wall_time();

    // ⭐ KEY FIX: Create thread pool ONCE for entire computation
    #pragma omp parallel
    {
        // All threads stay alive for entire sweep
        for (int sweep = 0; sweep < sweeps; ++sweep) {
            for (int q = 0; q < blocks - 1; ++q) {
                // Only master thread does serial work
                #pragma omp master
                {
                    for (int jj = 0; jj < k; ++jj) {
                        memcpy(Ubuf + (size_t)jj * m, A + (size_t)(q*k + jj) * m, sizeof(double)*m);
                    }
                }
                #pragma omp barrier  // Wait for master to finish
                
                for (int p = q + 1; p < blocks; ++p) {
                    #pragma omp master
                    {
                        for (int jj=0;jj<k;++jj)
                            memcpy(Ubuf + (size_t)(k + jj)*m, A + (size_t)(p*k + jj)*m, sizeof(double)*m);
                    }
                    #pragma omp barrier

                    int tk = 2*k;
                    
                    // Parallel GEMM (threads already exist)
                    dgemm_simple('T','N', tk, tk, m, 1.0, Ubuf, m, Ubuf, m, 0.0, G, tk);
                    #pragma omp barrier
                    
                    // Serial Jacobi (only master does this)
                    #pragma omp master
                    {
                        jacobi_eigen_small(G, R, tk, 200, 1e-12);
                    }
                    #pragma omp barrier
                    
                    // Parallel GEMM
                    dgemm_simple('N','N', m, tk, tk, 1.0, Ubuf, m, R, tk, 0.0, Utmp, m);
                    #pragma omp barrier
                    
                    #pragma omp master
                    {
                        memcpy(Ubuf, Utmp, sizeof(double)*m*tk);
                        
                        for (int jj=0;jj<k;++jj)
                            memcpy(Vsub + (size_t)jj*n, V + (size_t)(q*k + jj)*n, sizeof(double)*n);
                        for (int jj=0;jj<k;++jj)
                            memcpy(Vsub + (size_t)(k + jj)*n, V + (size_t)(p*k + jj)*n, sizeof(double)*n);
                    }
                    #pragma omp barrier
                    
                    // Parallel GEMM
                    dgemm_simple('N','N', n, tk, tk, 1.0, Vsub, n, R, tk, 0.0, Vtmp, n);
                    #pragma omp barrier
                    
                    #pragma omp master
                    {
                        for (int jj=0;jj<k;++jj)
                            memcpy(V + (size_t)(q*k + jj)*n, Vtmp + (size_t)jj*n, sizeof(double)*n);
                        for (int jj=0;jj<k;++jj)
                            memcpy(V + (size_t)(p*k + jj)*n, Vtmp + (size_t)(k + jj)*n, sizeof(double)*n);
                        for (int jj=0;jj<k;++jj)
                            memcpy(A + (size_t)(p*k + jj)*m, Ubuf + (size_t)(k + jj)*m, sizeof(double)*m);
                    }
                    #pragma omp barrier
                }
                
                #pragma omp master
                {
                    for (int jj=0;jj<k;++jj)
                        memcpy(A + (size_t)(q*k + jj)*m, Ubuf + (size_t)jj*m, sizeof(double)*m);
                }
                #pragma omp barrier
            }
            
            // Parallel normalization
            normalize_columns(A, m, n);
            #pragma omp barrier
        }
    } // Thread pool destroyed here

    double t1 = wall_time();
    printf("Elapsed time (BCV with V, OpenMP) = %.6f seconds\n", t1 - t0);

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
    return 0;
}