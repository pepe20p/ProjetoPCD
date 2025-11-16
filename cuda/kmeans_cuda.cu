/* kmeans_cuda.cu
   K-means 1D (C99), implementação "naive":
   - Lê X (N linhas, 1 coluna) e C_init (K linhas, 1 coluna) de CSVs sem cabeçalho.
   - Itera assignment + update até max_iter ou variação relativa do SSE < eps.
   - Salva (opcional) assign (N linhas) e centróides finais (K linhas).

   Compilar: nvcc -O2 kmeans.cu -arch=sm_86 -o means
   Uso:./kmeans_1d_naive dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

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

/* ---------- k-means 1D ---------- */
/* assignment: para cada X[i], encontra c com menor (X[i]-C[c])^2 */
__global__ void assignment_kernel(const double *X, const double *C, int *assign, double *sse, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; //Indice
    if (i < N) { //nao executar fora do indice
        int best = -1;
        double bestd = 1e300;
        for(int c=0;c<K;c++){
            double diff = X[i] - C[c];
            double d = diff*diff;
            if(d < bestd){ bestd = d; best = c; }
        }
        assign[i] = best;
        atomicAdd(sse, bestd);
    }
}
/* update: média dos pontos de cada cluster (1D)
   se cluster vazio, copia X[0] (estratégia naive) */
__global__ void update_sum_cnt_kernel(const double *X, const int *assign, double *sum, int *cnt, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; //Indice
    if (i < N) { //nao executar fora do indice
        int a = assign[i];
        atomicAdd(&sum[a], X[i]);
        atomicAdd(&cnt[a], 1);
    }
}

__global__ void update_calc_avg_kernel(double *C, const double *sum, const int *cnt, const double *X, int K) {
    int c = blockIdx.x * blockDim.x + threadIdx.x; //Indice
    if (c < K) { //nao executar fora do indice
        if (cnt[c] > 0) C[c] = sum[c] / (double)cnt[c];
        else            C[c] = X[0]; /* simples: cluster vazio recebe o primeiro ponto */
    }
}

int main(int argc, char **argv){
    clock_t t0 = clock(); //inicial
    if(argc < 3){
        printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]\n", argv[0]);
        printf("Obs: arquivos CSV com 1 coluna (1 valor por linha), sem cabeçalho.\n");
        return 1;
    }
    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc>3)? atoi(argv[3]) : 50;
    double eps   = (argc>4)? atof(argv[4]) : 1e-4;
    const char *outAssign   = (argc>5)? argv[5] : NULL;
    const char *outCentroid = (argc>6)? argv[6] : NULL;

    if(max_iter <= 0 || eps <= 0.0){
        fprintf(stderr,"Parâmetros inválidos: max_iter>0 e eps>0\n");
        return 1;
    }

    int N=0, K=0;
    double sse = 0.0;
    double *X = read_csv_1col(pathX, &N);
    double *C = read_csv_1col(pathC, &K);
    int *assign = (int*)malloc((size_t)N * sizeof(int));
    if(!assign){ fprintf(stderr,"Sem memoria para assign\n"); free(X); free(C); return 1; }

    double *gpu_X, *gpu_C, *gpu_sum, *gpu_sse;
    int *gpu_assign, *gpu_cnt;
    cudaMalloc(&gpu_X, (size_t)N * sizeof(double));
    cudaMalloc(&gpu_assign, (size_t)N * sizeof(int));
    cudaMalloc(&gpu_C, (size_t)K * sizeof(double));
    cudaMalloc(&gpu_sum, (size_t)K * sizeof(double));
    cudaMalloc(&gpu_cnt, (size_t)K * sizeof(int));
    cudaMalloc(&gpu_sse, sizeof(double));

    clock_t t1 = clock();
    cudaMemcpy(gpu_X, X, (size_t)N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_C, C, (size_t)K * sizeof(double), cudaMemcpyHostToDevice);
    
    clock_t t2 = clock(); //copia h2d
    
    int threadsPerBlock = 512;
    
    int blocosN = (N + threadsPerBlock - 1) / threadsPerBlock;
    int blocosK = (K + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("tempo h2d: %.f ms",(1000.0 * (double)(t2 - t1) / (double)CLOCKS_PER_SEC));
    printf(" %ld MB\n",(N * (sizeof(double)+sizeof(int)) + K * (sizeof(double)*2+sizeof(int))) / (1024*1024));
    double prev_sse = 1e300;
    int it;
    clock_t t5 = clock();
    for(it=0; it<max_iter; it++){
        cudaMemset(gpu_sse, 0, sizeof(double));
        assignment_kernel<<<blocosN, threadsPerBlock>>>(gpu_X, gpu_C, gpu_assign, gpu_sse, N, K);
        cudaMemcpy(&sse, gpu_sse, sizeof(double), cudaMemcpyDeviceToHost);
        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if(rel < eps){ it++; break; }
        prev_sse = sse;

        cudaMemset(gpu_sum, 0, (size_t)K * sizeof(double));
        cudaMemset(gpu_cnt, 0, (size_t)K * sizeof(int));
        update_sum_cnt_kernel<<<blocosN, threadsPerBlock>>>(gpu_X, gpu_assign, gpu_sum, gpu_cnt, N, K);
        update_calc_avg_kernel<<<blocosK, threadsPerBlock>>>(gpu_C, gpu_sum, gpu_cnt, gpu_X, K);
        cudaDeviceSynchronize();
    }    

    clock_t t3 = clock();
    printf("tempo kernel: %.f ms\n",(1000.0 * (double)(t3 - t5) / (double)CLOCKS_PER_SEC));
    cudaMemcpy(C, gpu_C, (size_t)K * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(assign,gpu_assign, (size_t)N*sizeof(int),cudaMemcpyDeviceToHost);
    
    clock_t t4 = clock();
    printf("tempo d2h: %.f ms",(1000.0 * (double)(t4 - t3) / (double)CLOCKS_PER_SEC));
    printf(" %ld MB\n",(N * (sizeof(double)+sizeof(int)) + K * (sizeof(double)*2+sizeof(int))) / (1024*1024));
    double ms = 1000.0 * (double)(t4 - t0) / (double)CLOCKS_PER_SEC;
    printf("K-means 1D (CUDA)\n");
    printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);
    printf("Iterações: %d | SSE final: %.6f | Tempo: %.1f ms\n", it, prev_sse, ms);
    
    write_assign_csv(outAssign, assign, N);
    write_centroids_csv(outCentroid, C, K);
    
    cudaFree(gpu_X);cudaFree(gpu_C);cudaFree(gpu_assign);cudaFree(gpu_sum);cudaFree(gpu_cnt);cudaFree(gpu_sse);
    free(X);
    free(C);
    free(assign);
    return 0;
}