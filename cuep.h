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

#ifndef CUEP_H
#define CUEP_H

/* Print runtime */
#define TIME_CALL

/* When the Doppler factor is to be considered zero */
#define SMALL_NUMBER 1.e-20

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

#ifdef __cplusplus
extern "C"{
#endif 

/* Structure to store device memory pointers */
struct cuep_plan {
    int N;
    int *dev_flag;
    double *dev_x;
    double *dev_Ep;
    double *dev_Em;
};

int cuep_create_plan(struct cuep_plan *d, int N);

int cuep_execute_plan(double *Em, double *Ep, int *flag, double *x, double *startpos, double *endpos, double *beta, double n, struct cuep_plan *d);

int cuep_destroy_plan(struct cuep_plan *d);

#ifdef __cplusplus
}
#endif

#endif /* CUEP_H */

