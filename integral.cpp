#include <iostream>
#include <cmath>
#include <omp.h>
#include <cstdlib>
#include <fstream>
using namespace std;

double f(double x) {
return exp(cos(x));
}

double trapezoidal(double a, double b, int n) {
double h = (b - a) / n;
double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
for (int i = 1; i < n; ++i) {
double x = a + i * h;
sum += f(x);
}
sum += (f(a) + f(b)) / 2.0;
return h * sum;
}
int main(int argc, char* argv[]) {
if (argc < 5) {
return 1;
}
double a = atof(argv[1]);
double b = atof(argv[2]);
int n = atoi(argv[3]);
int threads = atoi(argv[4]);
omp_set_num_threads(threads);
double start = omp_get_wtime();
double result = trapezoidal(a, b, n);
double end = omp_get_wtime();
double elapsed = end - start;
// Запис у файл
ofstream fout("result_" + to_string(threads) + ".txt");
fout << "threads,time,result\n";
fout << threads << "," << elapsed << "," << result << "\n";
fout.close();
return 0;
}
