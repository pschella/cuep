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

    const int N = 1e6;

    srand(time(NULL));

    n = 1.003;

    /* Allocate host memory */
    flag = new int[N];
    Ep = new double[3 * N];
    Em = new double[3 * N];
    x = new double[3 * N];

    /* Generate random particle */
    for (i=0; i<3; i++) {
        startpos[i] = 1000. * (double)rand() / RAND_MAX;
        endpos[i] = 1000. * (double)rand() / RAND_MAX;
        beta[i] = (double)rand() / RAND_MAX;
    }

    /* Generate random antenna positions */
    for (i=0; i<3*N; i++) {
        x[i] = 1000. * (double)rand() / RAND_MAX;
    }

    if (cuep_create_plan(&d, N) != 0) return 1;

    if (cuep_execute_plan(Em, Ep, flag, x, startpos, endpos, beta, n, &d) != 0) return 1;

    if (cuep_destroy_plan(&d) != 0) return 1;

    /* Print results */
    for (i=0; i<10; i++) {
      std::cout<<"flag "<<flag[i]<<" E+ "<<flag[i]<<" "<<Ep[3*i]<<" "<<Ep[3*i+1]<<" "<<Ep[3*i+2]<<" E- "<<Em[3*i]<<" "<<Em[3*i+1]<<" "<<Em[3*i+2]<<std::endl;
    }

    /* Cleanup */
    delete[] flag;
    delete[] Ep;
    delete[] Em;
    delete[] x;
}

