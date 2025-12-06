compilar
mpicc -O2 -std=c99 kmeans_1d_mpi.c -o kmeans_1d_mpi -lm

executar
mpirun -np 1 ./kmeans_1d_mpi dados.csv inicial.csv 1000 1e-6 assign.csv cent.csv