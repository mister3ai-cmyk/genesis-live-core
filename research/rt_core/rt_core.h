/* Business Source License 1.1 - см. research/LICENSE
 *
 * rt_core - нативное ядро «нервной системы» роя (S3, v5.0), C99.
 * Порт кинематического шлюза (research/swarm/kinematics.py, swarm_node
 * _traj_refutes) в 3D и геометрии Gr(k, N) (research/swarm/grassmann.py).
 *
 * Правила: -std=c99, только libc и libm, ни malloc/free - вся память в
 * rt_context_t (выделяет вызывающий: статически, на стеке или из Python
 * через rt_ctx_size()). Глобального изменяемого состояния нет.
 */
#ifndef RT_CORE_H
#define RT_CORE_H

#include <stddef.h>
#include <stdint.h>

#define RT_MAX_TARGETS 16      /* целей в контексте */
#define RT_RING        32      /* наблюдений на цель (кольцо, старейшее затирается) */
#define RT_MAX_AUTHORS 8       /* бортов-авторов заявок */
#define RT_GR_MAX_N    128     /* размерность объемлющего пространства для Gr(k, N) */
#define RT_GR_K        4       /* k по умолчанию */

/* коды возврата функций */
#define RT_OK          0
#define RT_NO_PRED     1       /* мало точек или авторов для прогноза */
#define RT_ERR_ARG    (-1)
#define RT_ERR_TIME   (-2)     /* немонотонная метка или метка из будущего */
#define RT_ERR_RANK   (-3)     /* вырожденный базис */

/* вердикты шлюза */
#define RT_PASS        0       /* заявке не возражаю */
#define RT_REFUTE      1       /* прогноз за мой датчик, против заявки - опровергаю */
#define RT_FAILSAFE    2       /* решения нет: работает базовое правило (v3.1) */

typedef struct {
    int64_t horizon_us;        /* окно заявок для прогноза, по умолчанию 4 с */
    int64_t max_skew_us;       /* допустимое расхождение меток времени, 50 мс */
    double  drop_tol;          /* допуск согласия точки с подгонкой, 400 */
    double  thr;               /* порог невязки шлюза, 800 */
    double  pos_tol;           /* допуск датчика: ближе - спора нет, 300 */
    int     min_pts;           /* минимум точек для прогноза, 4 */
} rt_cfg_t;

typedef struct {
    int64_t t_us;
    double  p[3];
    uint8_t author;
} rt_obs_t;

typedef struct {
    rt_obs_t obs[RT_RING];
    int      head;             /* куда писать следующее */
    int      count;
    int64_t  last_t[RT_MAX_AUTHORS];   /* последняя метка автора - монотонность */
} rt_ring_t;

typedef struct {
    rt_cfg_t  cfg;
    rt_ring_t ring[RT_MAX_TARGETS];
    uint32_t  n_push, n_reject_time;
    uint32_t  n_pass, n_refute, n_failsafe;
} rt_context_t;

size_t rt_ctx_size(void);
void   rt_cfg_default(rt_cfg_t *cfg);
int    rt_init(rt_context_t *ctx, const rt_cfg_t *cfg);   /* cfg == NULL - по умолчанию */

#endif
