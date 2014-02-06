#include <cuda_runtime.h>

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

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
const int threadsPerBlock = MIN(N, 1024);
const int numberOfBlocks = MIN((N+threadsPerBlock-1)/threadsPerBlock, 1024);

/* Constant (read only) device memory */
__constant__ double dev_startpos[3];
__constant__ double dev_endpos[3];
__constant__ double dev_beta[3];
__constant__ double dev_n;

__global__ void endpoint(int *flag, double *Ep, double *Em, double *x) {
    double r[3], rhat[3], R;
    double doppler, dot_product;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N) {
        flag[tid] = 0;
    
        /* Calculate E_- */
        r[0] = x[3*tid] - dev_startpos[0];
        r[1] = x[3*tid+1] - dev_startpos[1];
        r[2] = x[3*tid+2] - dev_startpos[2];
    
        R = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    
        rhat[0] = r[0] / R;
        rhat[1] = r[1] / R;
        rhat[2] = r[2] / R;
    
        dot_product = dev_beta[0]*rhat[0] + dev_beta[1]*rhat[1] + dev_beta[2]*rhat[2];
        doppler = (1. - dev_n * dot_product)*R;
    
        if (doppler < SMALL_NUMBER) {
            flag[tid] = 1;
            Em[3*tid] = 0.0;
            Em[3*tid+1] = 0.0;
            Em[3*tid+2] = 0.0;
        }
        else {
            Em[3*tid] = dot_product*rhat[0] - dev_beta[0];
            Em[3*tid+1] = dot_product*rhat[1] - dev_beta[1];
            Em[3*tid+2] = dot_product*rhat[2] - dev_beta[2];
        }
    
        /* Calculate E_+ */
        r[0] = x[3*tid] - dev_endpos[0];
        r[1] = x[3*tid+1] - dev_endpos[1];
        r[2] = x[3*tid+2] - dev_endpos[2];
    
        R = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    
        rhat[0] = r[0] / R;
        rhat[1] = r[1] / R;
        rhat[2] = r[2] / R;
    
        dot_product = dev_beta[0]*rhat[0] + dev_beta[1]*rhat[1] + dev_beta[2]*rhat[2];
        doppler = (1. - dev_n * dot_product)*R;
    
        if (doppler < SMALL_NUMBER) {
            Ep[3*tid] = 0.0;
            Ep[3*tid+1] = 0.0;
            Ep[3*tid+2] = 0.0;
        }
        else {
            flag[tid] = 0;
            Ep[3*tid] = dot_product*rhat[0] - dev_beta[0];
            Ep[3*tid+1] = dot_product*rhat[1] - dev_beta[1];
            Ep[3*tid+2] = dot_product*rhat[2] - dev_beta[2];
        }

        tid += blockDim.x * gridDim.x;
    }
}

int main(int argc, char* argv[])
{
    int i;

    int *flag, *dev_flag;

    double startpos[3];
    double endpos[3];
    double beta[3];
    double n;

    double *x, *Ep, *Em, *dev_x, *dev_Ep, *dev_Em;

    cudaError_t err;
    cudaEvent_t start, stop;

    float elapsed_time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    srand(time(NULL));

    /* Allocate host memory */
    flag = (int*)malloc(N * sizeof(int));
    Ep = (double*)malloc(3 * N * sizeof(double));
    Em = (double*)malloc(3 * N * sizeof(double));
    x = (double*)malloc(3 * N * sizeof(double));

    /* Allocate device memory */
    err = cudaMalloc(&dev_flag, N*sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc(&dev_Ep, 3*N*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc(&dev_Em, 3*N*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc(&dev_x, 3*N*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
        return 1;
    }

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

    /* Copy to device */
    err = cudaMemcpyToSymbol(dev_startpos, startpos, 3*sizeof(double));
    if (err != cudaSuccess) {
        printf("Error %s\n", cudaGetErrorString(err));
    }

    err = cudaMemcpyToSymbol(dev_endpos, endpos, 3*sizeof(double));
    if (err != cudaSuccess) {
        printf("Error %s\n", cudaGetErrorString(err));
    }

    err = cudaMemcpyToSymbol(dev_beta, beta, 3*sizeof(double));
    if (err != cudaSuccess) {
        printf("Error %s\n", cudaGetErrorString(err));
    }

    err = cudaMemcpyToSymbol(dev_n, &n, sizeof(double));
    if (err != cudaSuccess) {
        printf("Error %s\n", cudaGetErrorString(err));
    }

    err = cudaMemcpy(dev_x, x, 3*N*sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Error %s\n", cudaGetErrorString(err));
    }

    /* Call function on device, launching 1 thread block with N threads */
    cudaEventRecord(start, 0);
    endpoint<<<numberOfBlocks, threadsPerBlock>>>(dev_flag, dev_Ep, dev_Em, dev_x);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed_time, start, stop);

    /* Copy result back to host */
    err = cudaMemcpy(flag, dev_flag, N*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Error %s\n", cudaGetErrorString(err));
    }

    err = cudaMemcpy(Ep, dev_Ep, 3*N*sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Error %s\n", cudaGetErrorString(err));
    }

    err = cudaMemcpy(Em, dev_Em, 3*N*sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Error %s\n", cudaGetErrorString(err));
    }

    /* Print results */
    for (i=0; i<N; i++) {
        printf("flag %d E+ %.3f %.3f %.3f E- %.3f %.3f %.3f\n", flag[i], Ep[3*i], Ep[3*i+1], Ep[3*i+2], Em[3*i], Em[3*i+1], Em[3*i+2]);
    }

    printf("launched %d blocks with %d threads each\n", (N+threadsPerBlock-1)/threadsPerBlock, threadsPerBlock);
    printf("Runtime: %.3f ms\n", elapsed_time);

    /* Cleanup */
    free(flag);
    free(Ep);
    free(Em);
    free(x);

    cudaFree(dev_flag);
    cudaFree(dev_Ep);
    cudaFree(dev_Em);
    cudaFree(dev_x);
}

