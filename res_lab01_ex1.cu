#include <stdio.h>

void cpu_helloworld() { printf("Alô da CPU!\n"); }

__global__ void gpu_helloworld() {
    int threadId = threadIdx.x;
    printf("Alô da GPU! Meu threadId eh %d\n", threadId);
}

int main(int argc, char **argv) {
    dim3 grid(1);
    dim3 block(32);

    cpu_helloworld();

    gpu_helloworld<<<grid, block>>>();

    cudaDeviceSynchronize();

    return 0;
}
