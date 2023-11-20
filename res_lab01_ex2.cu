#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define BLOCK_SIZE 64
#define ARRAY_SIZE 128000

typedef struct timeval tval;

float generate_hash(int n, float *y) {
    float hash = 0.0f;

    for (int i = 0; i < n; i++) {
        hash += y[i];
    }

    return hash;
}

double get_elapsed(tval t0, tval t1) {
    return (double)(t1.tv_sec - t0.tv_sec) * 1000.0L +
           (double)(t1.tv_usec - t0.tv_usec) / 1000.0L;
}

void cpu_saxpy(int n, float a, float *x, float *y) {
    for (int i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

// TO-DO #2.1
// Declarando e implementando o kernel CUDA executará na GPU.
__global__ void gpu_saxpy(int n, float a, float *x, float *y) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        y[tid] = a * x[tid] + y[tid];
    }
}

int main(int argc, char **argv) {
    float a = 0.0f;
    float *x = NULL;
    float *y = NULL;
    float error = 0.0f;

    // Verifique se a contante foi fornecida
    // if (argc != 2) {
    //     fprintf(stderr, "Erro: A constante está faltando!\n");
    //     return -1;
    // }

    // Código original do problema SAXPY
    a = 10;
    x = (float *)malloc(sizeof(float) * ARRAY_SIZE);
    y = (float *)malloc(sizeof(float) * ARRAY_SIZE);
    for (int i = 0; i < ARRAY_SIZE; i++) {
        x[i] = 0.1f;
        y[i] = 0.2f;
    }

    // TO-DO #2.2
    // Definindo a distribuição das threads, em termos da dimensão da grade
    // (grid) e da dimensão de cada bloco (de threads) dentro da grade.
    dim3 grid((ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    // TO-DO #2.3.1
    // Declarando e definindo a memória GPU necessária para executar o kernel
    // CUDA.
    float *d_x, *d_y;
    cudaMalloc((void **)&d_x, sizeof(float) * ARRAY_SIZE);
    cudaMalloc((void **)&d_y, sizeof(float) * ARRAY_SIZE);

    // TO-DO #2.3.2
    // Transferindo os dados do host para a GPU.
    cudaMemcpy(d_x, x, sizeof(float) * ARRAY_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, sizeof(float) * ARRAY_SIZE, cudaMemcpyHostToDevice);

    // Variáveis para métricas de performance/tempo de execução da CPU.
    tval t_start, t_end;
    gettimeofday(&t_start, NULL);
    // Execução na CPU para fins de comparação
    cpu_saxpy(ARRAY_SIZE, a, x, y);
    gettimeofday(&t_end, NULL);
    double elapsed_cpu = get_elapsed(t_start, t_end);
    error = generate_hash(ARRAY_SIZE, y);

    printf("Execução na CPU terminada. Taxa de erro = %.6f.\n", error);
    printf("Tempo de execução na CPU: %.6f ms.\n", elapsed_cpu);

    // Variáveis para métricas de performance/tempo de execução da GPU.
    tval t_start_gpu, t_end_gpu;
    gettimeofday(&t_start_gpu, NULL);

    // TO-DO #2.4
    // Execute o kernel CUDA com os parâmetros correspondentes.
    gpu_saxpy<<<grid, block>>>(ARRAY_SIZE, a, d_x, d_y);
    cudaDeviceSynchronize();

    gettimeofday(&t_end_gpu, NULL);
    double elapsed_gpu = get_elapsed(t_start_gpu, t_end_gpu);

    // TO-DO #2.5.1
    // Transferindo os resultados da GPU para o host.
    cudaMemcpy(y, d_y, sizeof(float) * ARRAY_SIZE, cudaMemcpyDeviceToHost);

    error = fabsf(error - generate_hash(ARRAY_SIZE, y));

    printf("Execução na GPU terminada. Taxa de erro = %.6f.\n", error);
    printf("Tempo de execução na GPU: %.6f ms.\n", elapsed_gpu);

    if (error > 0.0001f) {
        fprintf(stderr, "Erro: a solução está incorreta!\n");
    }

    // Gerenciamento de memória.
    free(x);
    free(y);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}