/* Business Source License 1.1 - см. research/LICENSE
 *
 * Оценка задержки между двумя потоками датчиков взаимной корреляцией
 * (рассинхрон IMU и позы, дрейф часов узла). Без памяти: работает прямо по
 * входным массивам.
 */
#ifndef RT_XCORR_H
#define RT_XCORR_H

#include "rt_core.h"

#define RT_ERR_LOW_EXCITATION (-4)   /* в окне нет движения: оценка была бы шумом */

/* a, b - n отсчётов на общей сетке времени (шаг одинаковый). Для каждого
 * сдвига k в [-max_lag, max_lag] - коэффициент Пирсона по перекрытию
 * a[i] и b[i + k]; лучший k уточняется параболой по соседям.
 * lag > 0: b отстаёт от a на lag отсчётов (b[i + lag] ~ a[i]).
 * min_var - порог дисперсии каждого сигнала по окну: ниже - сигнала нет
 * (висение без вращения), RT_ERR_LOW_EXCITATION, а не случайный пик по шуму.
 * Требуется 1 <= max_lag <= n / 4. peak - корреляция в лучшей точке. */
int rt_xcorr_delay(const double *a, const double *b, int n, int max_lag,
                   double min_var, double *lag, double *peak);

/* То же для нескольких каналов (например, три компоненты угловой скорости):
 * a и b - channels подряд идущих массивов по n отсчётов. Ковариации и
 * дисперсии каналов суммируются; порог min_var - для суммарной дисперсии.
 * Модуль вектора теряет информацию (при постоянном вращении он почти
 * константа), компоненты - нет. */
int rt_xcorr_delay_multi(const double *a, const double *b, int n, int channels, int max_lag,
                         double min_var, double *lag, double *peak);

#endif
