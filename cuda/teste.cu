#include <stdio.h>
__global__ void teste() {
    printf("Block %d, Thread %d\n", blockIdx.x, threadIdx.x);
}
int main() {
    printf("Iniciando\n");
    teste<<<2, 4>>>();
    cudaDeviceSynchronize();
    printf("Finalizado.\n");
    return 0;
}