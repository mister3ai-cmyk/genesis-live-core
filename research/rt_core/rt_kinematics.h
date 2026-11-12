/* Business Source License 1.1 - см. research/LICENSE
 *
 * Кинематический шлюз v4.0 в 3D (SWARM_FINDINGS, Часть XI, v4.0-A3).
 * Времена - микросекунды общего времени узлов (GNSS/PTP): шлюз имеет
 * смысл только при синхронных часах (Часть XI, v4.0-A4).
 */
#ifndef RT_KINEMATICS_H
#define RT_KINEMATICS_H

#include "rt_core.h"

/* Заявка автора о цели в момент t_us; now_us - локальное время приёма.
 * RT_ERR_TIME: метка не больше прежней метки этого автора по этой цели
 * или опережает now_us больше чем на max_skew_us. */
int rt_kin_push(rt_context_t *ctx, int target, int author,
                int64_t t_us, int64_t now_us, const double p[3]);

/* Прогноз положения цели в момент t_us по заявкам за horizon_us до него,
 * кроме автора exclude_author (-1 - никого не исключать). Подгонка
 * многочленом (2-й степени при >= 6 точках, иначе 1-й), отбор согласия
 * «без одного автора»: выживает один лжец в окне.
 * RT_OK и out, или RT_NO_PRED. */
int rt_kin_predict(const rt_context_t *ctx, int target, int64_t t_us,
                   int exclude_author, double out[3]);

/* Правило шлюза: RT_REFUTE, только если мой датчик расходится с заявкой
 * больше pos_tol (max-норма), прогноз согласен с моим датчиком (<= thr) и не
 * согласен с заявкой (> thr). Метка заявки из будущего больше чем на
 * max_skew_us (рассинхрон часов), нет прогноза или мой датчик далёк от
 * прогноза - RT_FAILSAFE. Иначе RT_PASS. Задержка доставки (метка в
 * прошлом) шлюз не выключает.
 * res_claim (может быть NULL) - невязка заявки, -1 если прогноза не было. */
int rt_kin_gate(rt_context_t *ctx, int target, int author,
                int64_t t_claim_us, int64_t now_us,
                const double claim[3], const double mine[3], double *res_claim);

#endif
