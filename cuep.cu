#include <cuda_runtime.h>

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

/* Print runtime */
#define TIME_CALL

/* When the Doppler factor is to be considered zero */
#define SMALL_NUMBER 1.e-20

/* Number of antennas */
#define N 512

/* Scaling of random positions for antenna grid (arbitrary and not needed for actual code) */
#define SCALE 500

#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))

/* The number of parallel GPU threads is determined by the product of the number of
 * thread blocks launched and the number of threads per block. There are hardware limits
 * on both numbers. If possible a thread is launched for each observer position.
 * If the number of observer positions exceeds the maximum number of threads allowed per block
 * (which is usually 1024 but is hardware dependent) than multiple blocks are launched.
 * The maximum number of threads launched in total is 1024 (blocks) * 1024 (threads per block)
 * if even more antennas are needed all threads loop over several antennas instead.
 * For performance one might want to play with the 1024 and try different powers of two (within hardware limits).
 */

#define MAX_BLOCKS 1024
#define MAX_THREADS_PER_BLOCK 1024

const int threadsPerBlock = MIN(N, MAX_THREADS_PER_BLOCK);
const int numberOfBlocks = MIN((N+threadsPerBlock-1)/threadsPerBlock, MAX_BLOCKS);

/* Constant (read only) device memory */
__constant__ double dev_startpos[3];
__constant__ double dev_endpos[3];
__constant__ double dev_beta[3];
__constant__ double dev_n;

/* Structure to store device memory pointers */
struct cuep_plan {
    int *dev_flag;
    double *dev_x;
    double *dev_Ep;
    double *dev_Em;
};

__global__ void endpoint(int *dev_flag, double *dev_Em, double *dev_Ep, double *dev_x) {
    double r[3], rhat[3], R;
    double doppler, dot_product;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N) {
        dev_flag[tid] = 0;
    
        /* Calculate E_- */
        r[0] = dev_x[3*tid] - dev_startpos[0];
        r[1] = dev_x[3*tid+1] - dev_startpos[1];
        r[2] = dev_x[3*tid+2] - dev_startpos[2];
    
        R = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    
        rhat[0] = r[0] / R;
        rhat[1] = r[1] / R;
        rhat[2] = r[2] / R;
    
        dot_product = dev_beta[0]*rhat[0] + dev_beta[1]*rhat[1] + dev_beta[2]*rhat[2];
        doppler = (1. - dev_n * dot_product)*R;
    
        if (doppler < SMALL_NUMBER) {
            dev_flag[tid] = 1;
            dev_Em[3*tid] = 0.0;
            dev_Em[3*tid+1] = 0.0;
            dev_Em[3*tid+2] = 0.0;
        }
        else {
            dev_Em[3*tid] = dot_product*rhat[0] - dev_beta[0];
            dev_Em[3*tid+1] = dot_product*rhat[1] - dev_beta[1];
            dev_Em[3*tid+2] = dot_product*rhat[2] - dev_beta[2];
        }
    
        /* Calculate E_+ */
        r[0] = dev_x[3*tid] - dev_endpos[0];
        r[1] = dev_x[3*tid+1] - dev_endpos[1];
        r[2] = dev_x[3*tid+2] - dev_endpos[2];
    
        R = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    
        rhat[0] = r[0] / R;
        rhat[1] = r[1] / R;
        rhat[2] = r[2] / R;
    
        dot_product = dev_beta[0]*rhat[0] + dev_beta[1]*rhat[1] + dev_beta[2]*rhat[2];
        doppler = (1. - dev_n * dot_product)*R;
    
        if (doppler < SMALL_NUMBER) {
            dev_Ep[3*tid] = 0.0;
            dev_Ep[3*tid+1] = 0.0;
            dev_Ep[3*tid+2] = 0.0;
        }
        else {
            dev_flag[tid] = 0;
            dev_Ep[3*tid] = dot_product*rhat[0] - dev_beta[0];
            dev_Ep[3*tid+1] = dot_product*rhat[1] - dev_beta[1];
            dev_Ep[3*tid+2] = dot_product*rhat[2] - dev_beta[2];
        }

        tid += blockDim.x * gridDim.x;
    }
}

int cuep_create_plan(struct cuep_plan *d)
{
    cudaError_t err;

    /* Allocate device memory */
    err = cudaMalloc(&(d->dev_flag), N*sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc(&(d->dev_Ep), 3*N*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc(&(d->dev_Em), 3*N*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc(&(d->dev_x), 3*N*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}

int cuep_execute_plan(double *Em, double *Ep, int *flag, double *x, double *startpos, double *endpos, double *beta, double n, struct cuep_plan *d)
{
    cudaError_t err;
#ifdef TIME_CALL
    cudaEvent_t start, stop;

    float elapsed_time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#endif

    /* Copy to device */
    err = cudaMemcpyToSymbol(dev_startpos, startpos, 3*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpyToSymbol(dev_endpos, endpos, 3*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpyToSymbol(dev_beta, beta, 3*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpyToSymbol(dev_n, &n, sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(d->dev_x, x, 3*N*sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
        return 1;
    }

    /* Call function on device, launching 1 thread block with N threads */
#ifdef TIME_CALL
    cudaEventRecord(start, 0);
#endif
    endpoint<<<numberOfBlocks, threadsPerBlock>>>(d->dev_flag, d->dev_Em, d->dev_Ep, d->dev_x);
#ifdef TIME_CALL
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    printf("launched %d blocks with %d threads each\n", (N+threadsPerBlock-1)/threadsPerBlock, threadsPerBlock);
    printf("Runtime: %.3f ms\n", elapsed_time);
#endif

    /* Copy result back to host */
    err = cudaMemcpy(flag, d->dev_flag, N*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(Ep, d->dev_Ep, 3*N*sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(Em, d->dev_Em, 3*N*sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}

int cuep_destroy_plan(struct cuep_plan *d)
{
    cudaError_t err;

    cudaFree(d->dev_flag);
    cudaFree(d->dev_Ep);
    cudaFree(d->dev_Em);
    cudaFree(d->dev_x);

    if ((err = cudaGetLastError()) != cudaSuccess) {
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}

int main(int argc, char* argv[])
{
    int i;
    int *flag;
    double startpos[3];
    double endpos[3];
    double beta[3];
    double n;
    double *x, *Ep, *Em;
    struct cuep_plan d;

    srand(time(NULL));

    n = 1.003;

    /* Allocate host memory */
    flag = (int*)malloc(N * sizeof(int));
    Ep = (double*)malloc(3 * N * sizeof(double));
    Em = (double*)malloc(3 * N * sizeof(double));
    x = (double*)malloc(3 * N * sizeof(double));

    /* Generate random particle */
    for (i=0; i<3; i++) {
        startpos[i] = 1000. * (double)rand() / RAND_MAX;
        endpos[i] = 1000. * (double)rand() / RAND_MAX;
        beta[i] = (double)rand() / RAND_MAX;
    }

    /* Generate random antenna positions */
    for (i=0; i<3*N; i++) {
        x[i] = SCALE * (double)rand() / RAND_MAX;
    }

    if (cuep_create_plan(&d) != 0) return 1;

    if (cuep_execute_plan(Em, Ep, flag, x, startpos, endpos, beta, n, &d) != 0) return 1;

    if (cuep_destroy_plan(&d) != 0) return 1;

    /* Print results */
    for (i=0; i<10; i++) {
        printf("flag %d E+ %.3f %.3f %.3f E- %.3f %.3f %.3f\n", flag[i], Ep[3*i], Ep[3*i+1], Ep[3*i+2], Em[3*i], Em[3*i+1], Em[3*i+2]);
    }

    /* Cleanup */
    free(flag);
    free(Ep);
    free(Em);
    free(x);
}

