/* Business Source License 1.1 - см. research/LICENSE
 *
 * Gr(k, N): ортонормирование и хордальное расстояние.
 * Матрицы N x k хранятся по столбцам: элемент (i, j) = A[j * n + i].
 */
#ifndef RT_GRASSMANN_H
#define RT_GRASSMANN_H

#include "rt_core.h"

/* Модифицированный Грам-Шмидт с одним проходом реортогонализации.
 * A и Q - n x k по столбцам, могут совпадать (на месте).
 * RT_ERR_RANK, если столбец линейно зависим от предыдущих. */
int rt_gr_orth_mgs(int n, int k, const double *A, double *Q);

/* d^2 = k - ||U^T V||_F^2 = sum sin^2 главных углов, 0 <= d^2 <= k.
 * U, V - ортонормированные n x k по столбцам. */
double rt_gr_chordal2(int n, int k, const double *U, const double *V);

#endif
