#include <mpi.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

using namespace std;

double f(double x) {
    return exp(cos(x));
}

double local_trapezoidal(double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.0;
    for (int i = 1; i < n; ++i) {
        double x = a + i * h;
        sum += f(x);
    }
    sum += (f(a) + f(b)) / 2.0;
    return h * sum;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (rank == 0) {
            cerr << "Usage: " << argv[0] << " a b n\n";
        }
        MPI_Finalize();
        return 1;
    }

    double a = atof(argv[1]);
    double b = atof(argv[2]);
    int n = atoi(argv[3]);

    double h = (b - a) / n;
    int local_n = n / size;
    double local_a = a + rank * local_n * h;
    double local_b = local_a + local_n * h;

    double start = MPI_Wtime();
    double local_result = local_trapezoidal(local_a, local_b, local_n);
    double global_result = 0.0;

    MPI_Reduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    double end = MPI_Wtime();

    if (rank == 0) {
        double elapsed = end - start;
        ofstream fout("result_" + to_string(size) + ".txt");
        fout << "Processes:"<< size << "\n";
        fout << "Time:"<< elapsed << "\n";
        fout << "Result:"<< global_result << "\n";
        fout.close(); 
    }

    MPI_Finalize();
    return 0;
}
