// corrected_bcv_simple.c  -- minimal fixes (contiguous column-major, offset loading, one local sweep)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h> // for memset

double wall_time() {
    struct timeval tv; gettimeofday(&tv,NULL); return tv.tv_sec + tv.tv_usec*1e-6;
}

double *allocate_mat_colmajor(int m, int n) {
    double *M = NULL;
    size_t bytes = sizeof(double) * (size_t)m * (size_t)n;
    if (posix_memalign((void**)&M, 64, bytes) != 0) return NULL;
    /* zero initialize to avoid reading uninitialized memory */
    memset(M, 0, bytes);
    return M;
}
#define A_at(A,m,i,j) ((A)[ (size_t)(j)*(m) + (i) ])  // column-major

double column_norm_sq_col(const double *A, int m, int col) {
    double s = 0.0;
    const double *colptr = A + (size_t)col * m;
    for (int i=0;i<m;++i){ double v = colptr[i]; s += v*v; }
    return s;
}
double column_dot_col(const double *A, int m, int c1, int c2) {
    double s = 0.0;
    const double *p1 = A + (size_t)c1 * m, *p2 = A + (size_t)c2 * m;
    for (int i=0;i<m;++i) s += p1[i] * p2[i];
    return s;
}
void scale_column_col(double *A, int m, int col, double scale) {
    double *p = A + (size_t)col*m;
    for (int i=0;i<m;++i) p[i] *= scale;
}

void load_submatrix_offset(double *U, const double *A, int m, int n,
                           int start_col, int k, int col_offset, int two_k) {
    // U stored column-major with width two_k
    for (int j=0;j<k;++j){
        const double *src = A + (size_t)(start_col + j) * m;
        double *dst = U + (size_t)(col_offset + j) * m;
        for (int i=0;i<m;++i) dst[i] = src[i];
    }
}
void store_submatrix_offset(double *A, const double *U, int m, int n,
                            int start_col, int k, int col_offset, int two_k) {
    for (int j=0;j<k;++j){
        const double *src = U + (size_t)(col_offset + j)*m;
        double *dst = A + (size_t)(start_col + j)*m;
        for (int i=0;i<m;++i) dst[i] = src[i];
    }
}

// Basic Givens sweep on U (2k columns, column-major)
void givens_rotation_local(double *U, int m, int two_k) {
    for (int i = 0; i < two_k - 1; ++i) {
        for (int j = i + 1; j < two_k; ++j) {
            double alpha = 0.0, beta = 0.0, gamma = 0.0;
            double *pi = U + (size_t)i*m, *pj = U + (size_t)j*m;
            for (int r = 0; r < m; ++r) {
                double ui = pi[r], uj = pj[r];
                alpha += ui*ui; beta += uj*uj; gamma += ui*uj;
            }
            if (fabs(gamma) < 1e-14) continue;
            double tau = (beta - alpha) / (2.0 * gamma);
            double t = (tau >= 0.0 ? 1.0 : -1.0) / (fabs(tau) + sqrt(1.0 + tau*tau));
            double c = 1.0 / sqrt(1.0 + t*t);
            double s = t * c;
            for (int r = 0; r < m; ++r) {
                double ui = pi[r], uj = pj[r];
                pi[r] = c*ui - s*uj;
                pj[r] = s*ui + c*uj;
            }
        }
    }
}

int main(int argc, char **argv){
    int m=2000,n=2000,k=20,sweeps=5;
    if (argc>=5){ m=atoi(argv[1]); n=atoi(argv[2]); k=atoi(argv[3]); sweeps=atoi(argv[4]); }
    if (n % k != 0){ fprintf(stderr,"n must be divisible by k\n"); return 1; }
    printf("BCV-Jacobi (no V) %dx%d k=%d sweeps=%d\n",m,n,k,sweeps);

    double *A = allocate_mat_colmajor(m,n);
    if (A == NULL) { fprintf(stderr,"Allocation A failed\n"); return 1; }

    for (int j=0;j<n;++j) for (int i=0;i<m;++i) A_at(A,m,i,j) = sin((double)(i+1))*cos((double)(j+1)) + ((i+j)%7)*0.01;

    int two_k = 2*k;
    double *U = allocate_mat_colmajor(m, two_k);
    if (U == NULL) { fprintf(stderr,"Allocation U failed\n"); free(A); return 1; }

    int blocks = n / k;
    double t0 = wall_time();
    for (int sweep=0; sweep<sweeps; ++sweep) {
        for (int q=0; q<blocks-1; ++q) {
            load_submatrix_offset(U, A, m, n, q*k, k, 0, two_k);
            for (int p=q+1; p<blocks; ++p) {
                load_submatrix_offset(U, A, m, n, p*k, k, k, two_k); // load second block into offset k
                // single local sweep (you can do more iterations if desired)
                givens_rotation_local(U, m, two_k);
                store_submatrix_offset(A, U, m, n, p*k, k, k, two_k);
            }
            store_submatrix_offset(A, U, m, n, q*k, k, 0, two_k);
        }
        // normalize columns
        for (int col=0; col<n; ++col) {
            double nrm = sqrt(column_norm_sq_col(A, m, col));
            if (nrm > 1e-14) scale_column_col(A, m, col, 1.0 / nrm);
        }
    }
    double t1 = wall_time();
    printf("Elapsed time(BCV only) = %.6f s\n", t1 - t0);

    free(A); free(U);
    return 0;
}