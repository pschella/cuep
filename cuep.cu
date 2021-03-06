/*
Copyright (C) 2014 Pim Schellart <P.Schellart@astro.ru.nl>

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/

#include <cuda_runtime.h>

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#include "cuep.h"

#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))

/* Constant (read only) device memory */
__constant__ double dev_startpos[3];
__constant__ double dev_endpos[3];
__constant__ double dev_beta[3];
__constant__ double dev_n;
__constant__ int dev_N;

__global__ void endpoint(int *dev_flag, double *dev_Em, double *dev_Ep, double *dev_x) {
    double r[3], rhat[3], R;
    double doppler, dot_product;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < dev_N) {
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
            dev_Em[3*tid] = (dot_product*rhat[0] - dev_beta[0]) / doppler;
            dev_Em[3*tid+1] = (dot_product*rhat[1] - dev_beta[1]) / doppler;
            dev_Em[3*tid+2] = (dot_product*rhat[2] - dev_beta[2]) / doppler;
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
            dev_Ep[3*tid] = (dot_product*rhat[0] - dev_beta[0]) / doppler;
            dev_Ep[3*tid+1] = (dot_product*rhat[1] - dev_beta[1]) / doppler;
            dev_Ep[3*tid+2] = (dot_product*rhat[2] - dev_beta[2]) / doppler;
        }

        tid += blockDim.x * gridDim.x;
    }
}

int cuep_create_plan(struct cuep_plan *d, int N)
{
    cudaError_t err;

    d->N = N;

    /* Allocate device memory */
    err = cudaMalloc(&(d->dev_flag), d->N*sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc(&(d->dev_Ep), 3*d->N*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc(&(d->dev_Em), 3*d->N*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc(&(d->dev_x), 3*d->N*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}

int cuep_execute_plan(double *Em, double *Ep, int *flag, double *x, double *startpos, double *endpos, double *beta, double n, struct cuep_plan *d)
{
    const int threadsPerBlock = MIN(d->N, MAX_THREADS_PER_BLOCK);
    const int numberOfBlocks = MIN((d->N+threadsPerBlock-1)/threadsPerBlock, MAX_BLOCKS);

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

    err = cudaMemcpyToSymbol(dev_N, &(d->N), sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(d->dev_x, x, 3*d->N*sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
        return 1;
    }

    /* Call function on device */
#ifdef TIME_CALL
    cudaEventRecord(start, 0);
#endif
    endpoint<<<numberOfBlocks, threadsPerBlock>>>(d->dev_flag, d->dev_Em, d->dev_Ep, d->dev_x);
#ifdef TIME_CALL
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    printf("launched %d blocks with %d threads each\n", numberOfBlocks, threadsPerBlock);
    printf("Runtime: %.3f ms\n", elapsed_time);
#endif

    /* Copy result back to host */
    err = cudaMemcpy(flag, d->dev_flag, d->N*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(Ep, d->dev_Ep, 3*d->N*sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(Em, d->dev_Em, 3*d->N*sizeof(double), cudaMemcpyDeviceToHost);
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

