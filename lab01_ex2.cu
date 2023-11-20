
#include <stdio.h>
#include <sys/time.h>

#define BLOCK_SIZE 256
#define ARRAY_SIZE 16777216

typedef struct timeval tval;

/**
 * Método para gerar um "hash".
 */
float generate_hash(int n, float *y)
{
    float hash = 0.0f;

    for (int i = 0; i < n; i++)
    {
        hash += y[i];
    }

    return hash;
}

/**
 * Método que calcula o tempo entre dois intervalos de tempo (in millisegundos).
 */
double get_elapsed(tval t0, tval t1)
{
    return (double)(t1.tv_sec - t0.tv_sec) * 1000.0L + (double)(t1.tv_usec - t0.tv_usec) / 1000.0L;
}

/**
 * Implementação SAXPY usando a CPU.
 */
void cpu_saxpy(int n, float a, float *x, float *y)
{
    for (int i = 0; i < n; i++)
    {
        y[i] = a * x[i] + y[i];
    }
}

////////////////
// TO-DO #2.1 /////////////////////////////////////////////////////////////
// Declare o kernel gpu_saxpy() com a mesma interface da cpu_saxpy() //
///////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    float a     = 0.0f;
    float *x    = NULL;
    float *y    = NULL;
    float error = 0.0f;

    ////////////////
    // TO-DO #2.2 ///////////////////////////////
    // Defina a grade(grid) e bloco            //
    /////////////////////////////////////////////

    //////////////////
    // TO-DO #2.3.1 ////////////////////////////////
    // Declare os ponteiros da device d_x and d_y //
    ////////////////////////////////////////////////

    // Verifique se a contante foi fornecida
    if (argc != 2)
    {
        fprintf(stderr, "Erro: A constante está faltando!\n");
        return -1;
    }

    // Obtenha a constante e aloque os arrays na CPU
    a = atof(argv[1]);
    x = (float *)malloc(sizeof(float) * ARRAY_SIZE);
    y = (float *)malloc(sizeof(float) * ARRAY_SIZE);

    // Inicialize-os com valores fixos
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        x[i] = 0.1f;
        y[i] = 0.2f;
    }

    //////////////////
    // TO-DO #2.3.2 ////////////////////////////////////////////////////////
    // Aloque d_x e d_y na GPU, e copie o conteúdo a partir da CPU        //
    ////////////////////////////////////////////////////////////////////////

    // Execute o código da CPU
    cpu_saxpy(ARRAY_SIZE, a, x, y);

    // Calcule o "hash" do resultado da CPU
    error = generate_hash(ARRAY_SIZE, y);

    ////////////////
    // TO-DO #2.4 ////////////////////////////////////////
    // Chame o kernel GPU gpu_saxpy() com d_x e d_y     //
    //////////////////////////////////////////////////////

    //////////////////
    // TO-DO #2.5.1 ////////////////////////////////////////////////////
    // Copie o conteúdo de d_y da GPU para o array y na CPU           //
    ////////////////////////////////////////////////////////////////////

    // Calcule o "hash" do resultado da GPU
    error = fabsf(error - generate_hash(ARRAY_SIZE, y));

    // Confirme que a execução terminou
    printf("Execução terminada (error=%.6f).\n", error);

    if (error > 0.0001f)
    {
        fprintf(stderr, "Erro: a solução está incorreta!\n");
    }

    // Libere todas as alocações
    free(x);
    free(y);

    //////////////////
    // TO-DO #2.5.2 /////////
    // Libere d_x e d_y    //
    /////////////////////////

    return 0;
}

