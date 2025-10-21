compilar
gcc -O2 -fopenmp -std=c99 kmeans_1d_omp.c -o kmeans_1d_omp -lm

executar
dados.csv inicial.csv 1000 1e-6 assign.csv cent.csv