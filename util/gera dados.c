#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SEED 156322

double rand_double(double max_val) {
    return ((double)rand() / (double)RAND_MAX) * max_val;
}

int main(int argc, char *argv[]) {  // Argumentos [arquivo.csv] [quantidade] [valor_maximo]
    if (argc != 4) {
        printf("Erro argumento\n");
        return 1;
    }
    const char* filename = argv[1];
    long qtd = atol(argv[2]);
    double max = atof(argv[3]);
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Erro ao abrir o arquivo\n");
        return 1;
    }
    srand(SEED); 
    for (long i = 0; i < qtd; i++) {
        double random_value = rand_double(max);
        fprintf(file, "%.6f\n", random_value);
    }
    fclose(file);
    return 0;
}