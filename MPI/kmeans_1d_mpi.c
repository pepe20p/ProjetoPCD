/* kmeans_1d_mpi.c
K-means 1D (C99), implementação "mpi":
   - Lê X (N linhas, 1 coluna) e C_init (K linhas, 1 coluna) de CSVs sem cabeçalho.
   - Itera assignment + update até max_iter ou variação relativa do SSE < eps.
   - Salva (opcional) assign (N linhas) e centróides finais (K linhas).
   
   Compilar: mpicc -O2 -std=c99 kmeans_1d_mpi.c -o kmeans_1d_mpi -lm
   Executar: mpirun -np 4 ./kmeans_1d_mpi dados.csv centroides.csv
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h> // Biblioteca MPI

/* ---------- util CSV 1D: cada linha tem 1 número ---------- */
static int count_rows(const char *path){
    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); exit(1); }
    int rows=0; char line[8192];
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(!only_ws) rows++;
    }
    fclose(f);
    return rows;
}

static double *read_csv_1col(const char *path, int *n_out){
    int R = count_rows(path);
    if(R<=0){ fprintf(stderr,"Arquivo vazio: %s\n", path); exit(1); }
    double *A = (double*)malloc((size_t)R * sizeof(double));
    if(!A){ fprintf(stderr,"Sem memoria para %d linhas\n", R); exit(1); }

    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); free(A); exit(1); }

    char line[8192];
    int r=0;
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(only_ws) continue;
        
        /* aceita vírgula/ponto-e-vírgula/espaco/tab, pega o primeiro token numérico */
        const char *delim = ",; \t";
        char *tok = strtok(line, delim);
        if(!tok){ fprintf(stderr,"Linha %d sem valor em %s\n", r+1, path); free(A); fclose(f); exit(1); }
        A[r] = atof(tok);
        r++;
        if(r>R) break;
    }
    fclose(f);
    *n_out = R;
    return A;
}

static void write_assign_csv(const char *path, const int *assign, int N){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int i=0;i<N;i++) fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

static void write_centroids_csv(const char *path, const double *C, int K){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int c=0;c<K;c++) fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(argc < 3){
        if(rank == 0) {
            printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]\n", argv[0]);
            printf("Obs: arquivos CSV com 1 coluna (1 valor por linha), sem cabeçalho.\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc>3)? atoi(argv[3]) : 50;
    double eps   = (argc>4)? atof(argv[4]) : 1e-4;
    const char *outAssign = (argc>5)? argv[5] : NULL;
    const char *outCentroid = (argc>6)? argv[6] : NULL;

    double *X_global = NULL;
    double *C_global = NULL;
    int N_global = 0;
    int K = 0;

    if(rank == 0){
        X_global = read_csv_1col(pathX, &N_global);
        C_global = read_csv_1col(pathC, &K);
    }

    MPI_Bcast(&N_global, 1, MPI_INT, 0, MPI_COMM_WORLD); //transmite N
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD); //transmite K

    //separa dados para processos
    int *sendcounts = NULL;
    int *displs = NULL;
    int local_N = 0;
    if (rank == 0) {
        sendcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        int remainder = N_global % size;
        int sum = 0;
        for (int i = 0; i < size; i++) {
            sendcounts[i] = N_global / size;
            if (i < remainder) sendcounts[i]++;
            displs[i] = sum;
            sum += sendcounts[i];
        }
    }    

    //pontos
    MPI_Scatter(sendcounts, 1, MPI_INT, &local_N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    double *local_X = (double*)malloc(local_N * sizeof(double));
    int *local_assign = (int*)malloc(local_N * sizeof(int));
    double *C = (double*)malloc(K * sizeof(double));
    MPI_Scatterv(X_global, sendcounts, displs, MPI_DOUBLE, local_X, local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    //centroides
    if(rank == 0) memcpy(C, C_global, K * sizeof(double));
    MPI_Bcast(C, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //algoritimo
    double *local_sum = (double*)malloc(K * sizeof(double));
    int *local_cnt = (int*)malloc(K * sizeof(int));
    double *global_sum = (double*)malloc(K * sizeof(double));
    int *global_cnt = (int*)malloc(K * sizeof(int));
    int it;
    double start_time = MPI_Wtime();
    double global_sse = 0.0;
    double prev_sse = 1e300;

    for(it = 0; it < max_iter; it++) {
        //assign
        double local_sse = 0.0;
        for(int k=0; k<K; k++) { local_sum[k] = 0.0; local_cnt[k] = 0; }
        for(int i = 0; i < local_N; i++) {
            double bestd = 1e300;
            int best = -1;
            for(int c = 0; c < K; c++) {
                double diff = local_X[i] - C[c];
                double d = diff * diff;
                if(d < bestd) { bestd = d; best = c; }
            }
            local_assign[i] = best;
            local_sse += bestd;
            local_sum[best] += local_X[i];
            local_cnt[best]++;
        }

        //Update
        MPI_Allreduce(&local_sse, &global_sse, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); //sse
        MPI_Allreduce(local_sum, global_sum, K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); //sum
        MPI_Allreduce(local_cnt, global_cnt, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD); //cnt
        for(int k=0; k<K; k++) if(global_cnt[k] > 0) C[k] = global_sum[k] / (double)global_cnt[k];

        //condicao de parada
        double rel = fabs(global_sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if(rel < eps) { 
            it++; 
            break; 
        }
        prev_sse = global_sse;
    }
    double end_time = MPI_Wtime();

    //exibicao    
    if(rank == 0) {
        double time_ms = (end_time - start_time) * 1000.0;
        printf("K-means 1D (MPI) - %d Processos\n", size);
        printf("N=%d K=%d max_iter=%d eps=%g\n", N_global, K, max_iter, eps);
        printf("Iterações: %d | SSE final: %.6f | Tempo: %.1f ms\n", it, global_sse, time_ms);
        write_centroids_csv(outCentroid, C, K);
    }
    int *global_assign = NULL;
    if(rank == 0) global_assign = malloc(N_global * sizeof(int));
    MPI_Gatherv(local_assign, local_N, MPI_INT,global_assign, sendcounts, displs, MPI_INT,0, MPI_COMM_WORLD);
    if(rank == 0) {
        write_assign_csv(outAssign, global_assign, N_global);
        free(global_assign);
    }

    // Limpeza
    if(rank == 0) {
        free(X_global);
        free(sendcounts);
        free(displs);
    }
    free(local_X);
    free(local_assign);
    free(C);
    free(local_sum); 
    free(local_cnt);
    free(global_sum); 
    free(global_cnt);
    MPI_Finalize();
    return 0;
}