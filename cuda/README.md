compilar
nvcc -O2 kmeans_cuda.cu -arch=sm_86 -o means

especificar arquitetura para suportar atomic (3070-Ampere)

executar
dados.csv inicial.csv 1000 1e-6 assign.csv cent.csv