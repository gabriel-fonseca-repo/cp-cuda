#include <stdio.h>

/**
 * versão CPU !
 */
void cpu_helloworld() { printf("Alô da CPU!\n"); }

/**
 * versão GPU !
 */
__global__ void gpu_helloworld() {
    int threadId = threadIdx.x;
    printf("Alô da GPU! Meu threadId eh %d\n", threadId);
}

int main(int argc, char **argv) {
    dim3 grid(1);    // 1 bloco na grade
    dim3 block(32);  // 32 threads por bloco

    // Chama versão CPU
    cpu_helloworld();

    // Chama versão GPU
    gpu_helloworld<<<grid, block>>>();

    ////////////////
    // TO-DO #1.2 ////////////////////
    // Coloque suas alterações aqui! //
    //////////////////////////////////

    return 0;
}
