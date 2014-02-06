#include <iostream>
#include <cstdlib>

#include "cuep.h"

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
      std::cout<<"flag "<<flag[i]<<" E+ "<<flag[i]<<" "<<Ep[3*i]<<" "<<Ep[3*i+1]<<" "<<Ep[3*i+2]<<" E- "<<Em[3*i]<<" "<<Em[3*i+1]<<" "<<Em[3*i+2]<<std::endl;
    }

    /* Cleanup */
    free(flag);
    free(Ep);
    free(Em);
    free(x);
}

