/* Business Source License 1.1 - см. research/LICENSE
 *
 * Переносимость rt_core на встраиваемые цели (-ffreestanding, без libc и
 * libm). Обнуление и копирование - свои циклы; fabs и sqrt - встроенные
 * функции компилятора. RT_NO_LIBM: sqrt по Ньютону (для целей, где
 * __builtin_sqrt(double) не разворачивается в инструкцию, например Cortex-M4F
 * с FPU одинарной точности). Встраиваемые сборки добавляют
 * -fno-tree-loop-distribute-patterns, чтобы компилятор не превратил циклы
 * обратно в вызовы memset/memcpy.
 */
#ifndef RT_PORT_H
#define RT_PORT_H

#include <stddef.h>
#include <stdint.h>

static inline void rt_zero(void *p, size_t n)
{
    unsigned char *c = (unsigned char *)p;
    size_t i;
    for (i = 0; i < n; i++)
        c[i] = 0;
}

static inline void rt_copy(void *d, const void *s, size_t n)
{
    unsigned char *dc = (unsigned char *)d;
    const unsigned char *sc = (const unsigned char *)s;
    size_t i;
    for (i = 0; i < n; i++)
        dc[i] = sc[i];
}

#define RT_FABS(x) __builtin_fabs(x)

/* номер младшего установленного бита, x != 0 (Cortex-M: rbit + clz) */
#define RT_CTZ32(x) __builtin_ctz((unsigned int)(x))

/* sqrt по Ньютону: начальное приближение - половина показателя степени из
 * битов IEEE 754, затем 6 итераций (квадратичная сходимость, ~1 ulp).
 * x <= 0 -> 0; бесконечность и NaN возвращаются как есть. */
static inline double rt_sqrt_newton(double x)
{
    union { double d; uint64_t u; } v;
    double y;
    int i;
    if (!(x > 0.0))
        return x == x ? 0.0 : x;            /* 0, отрицательные -> 0; NaN -> NaN */
    if (x > 1.7976931348623157e308)
        return x;                            /* +inf */
    v.d = x;
    v.u = (v.u >> 1) + ((uint64_t)1023 << 51);
    y = v.d;
    for (i = 0; i < 6; i++)
        y = 0.5 * (y + x / y);
    return y;
}

#ifdef RT_NO_LIBM
#define RT_SQRT(x) rt_sqrt_newton(x)
#else
#define RT_SQRT(x) __builtin_sqrt(x)
#endif

#endif
