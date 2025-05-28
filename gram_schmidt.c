#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 1000
#define M 500

// Прототипи функцій
double vector_norm(double *v, int len);
double dot_product(double *a, double *b, int len);
double *A_column(double **A, int col, int n);

void generate_matrix(double **A, int rows, int cols, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            A[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

double vector_norm(double *v, int len) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < len; i++) {
        sum += v[i] * v[i];
    }
    return sqrt(sum);
}

double dot_product(double *a, double *b, int len) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < len; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Функція для виділення колонки матриці у вектор
double *A_column(double **A, int col, int n) {
    double *v = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) v[i] = A[i][col];
    return v;
}

// Класичний Грама-Шмідт
void gram_schmidt(double **A, double **Q, double **R, int n, int m) {
    for (int k = 0; k < m; k++) {
        for (int i = 0; i < n; i++)
            Q[i][k] = A[i][k];

        for (int j = 0; j < k; j++) {
            double *qj = A_column(Q, j, n);
            double *ak = A_column(A, k, n);
            R[j][k] = dot_product(qj, ak, n);
            free(ak);

            #pragma omp parallel for
            for (int i = 0; i < n; i++) {
                Q[i][k] -= R[j][k] * qj[i];
            }
            free(qj);
        }

        double *qk = A_column(Q, k, n);
        R[k][k] = vector_norm(qk, n);
        free(qk);

        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            Q[i][k] /= R[k][k];
        }
    }
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
