// gram_schmidt.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 1000 // кількість рядків
#define M 500  // кількість стовпців

// ініціалізація випадкової матриці з контрольованим seed
void generate_matrix(double **A, int rows, int cols, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            A[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

// норма вектора
double vector_norm(double *v, int len) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < len; i++) {
        sum += v[i] * v[i];
    }
    return sqrt(sum);
}

// добуток двох векторів
double dot_product(double *a, double *b, int len) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < len; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// QR-розклад класичним методом Грама-Шмідта
void gram_schmidt(double **A, double **Q, double **R, int n, int m) {
    for (int k = 0; k < m; k++) {
        for (int i = 0; i < n; i++)
            Q[i][k] = A[i][k];
        
        for (int j = 0; j < k; j++) {
            R[j][k] = dot_product(Q_column(Q, j, n), A_column(A, k, n), n);
            #pragma omp parallel for
            for (int i = 0; i < n; i++) {
                Q[i][k] -= R[j][k] * Q[i][j];
            }
        }

        R[k][k] = vector_norm(Q_column(Q, k, n), n);
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            Q[i][k] /= R[k][k];
        }
    }
}

double *A_column(double **A, int col, int n) {
    double *v = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) v[i] = A[i][col];
    return v;
}

double *Q_column(double **Q, int col, int n) {
    return A_column(Q, col, n); // те саме
}

double **alloc_matrix(int rows, int cols) {
    double **matrix = malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++)
        matrix[i] = malloc(cols * sizeof(double));
    return matrix;
}

void free_matrix(double **matrix, int rows) {
    for (int i = 0; i < rows; i++)
        free(matrix[i]);
    free(matrix);
}

int main(int argc, char *argv[]) {
    int num_threads = 1;
    if (argc > 1)
        num_threads = atoi(argv[1]);
    omp_set_num_threads(num_threads);

    double **A = alloc_matrix(N, M);
    double **Q = alloc_matrix(N, M);
    double **R = alloc_matrix(M, M);

    generate_matrix(A, N, M, 42); // однаковий seed

    double start = omp_get_wtime();
    gram_schmidt(A, Q, R, N, M);
    double end = omp_get_wtime();

    printf("Time: %lf seconds using %d threads\n", end - start, num_threads);

    free_matrix(A, N);
    free_matrix(Q, N);
    free_matrix(R, M);
    return 0;
}
