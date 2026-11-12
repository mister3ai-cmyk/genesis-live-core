/* Business Source License 1.1 - см. research/LICENSE
 *
 * Минимальный старт Cortex-M4/M7 без ОС для прогона tests/test_rt.c под
 * qemu-system-arm (mps2-an386 / mps2-an500): таблица векторов, копирование
 * .data, обнуление .bss, включение FPU, «водяной знак» стека, main(), выход
 * через semihosting. Вывод printf - newlib + librdimon (semihosting).
 * Ядро rt_core от этого файла не зависит.
 */
#include <stdint.h>
#include <stdio.h>

extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _sstack, _estack;
extern int main(void);
extern void initialise_monitor_handles(void);

void Reset_Handler(void);
void Fault_Handler(void);

#define STACK_MARK 0xA5A5A5A5u
#define SYS_EXIT   0x18
#define ADP_APP_EXIT     0x20026     /* qemu: код выхода 0 */
#define ADP_RUNTIME_ERR  0x20023     /* qemu: код выхода 1 */

__attribute__((section(".isr_vector"), used))
void (*const vectors[16])(void) = {
    (void (*)(void))&_estack, Reset_Handler,
    Fault_Handler, Fault_Handler, Fault_Handler, Fault_Handler, Fault_Handler,   /* NMI, Hard, MM, Bus, Usage */
    0, 0, 0, 0, Fault_Handler, Fault_Handler, 0, Fault_Handler, Fault_Handler    /* SVC, Debug, PendSV, SysTick */
};

static void semihost_exit(int failed)
{
    register int r0 __asm__("r0") = SYS_EXIT;
    register int r1 __asm__("r1") = failed ? ADP_RUNTIME_ERR : ADP_APP_EXIT;
    __asm__ volatile("bkpt 0xAB" : "+r"(r0) : "r"(r1) : "memory");
    for (;;)
        ;
}

void Fault_Handler(void)
{
    printf("FAULT: аппаратное исключение\n");
    semihost_exit(1);
}

void Reset_Handler(void)
{
    uint32_t *s = &_sidata, *d = &_sdata, *p, sp;
    int rc;
    while (d < &_edata)
        *d++ = *s++;
    for (d = &_sbss; d < &_ebss; d++)
        *d = 0;
    *(volatile uint32_t *)0xE000ED88 |= 0xFu << 20;     /* CPACR: CP10, CP11 - полный доступ к FPU */
    __asm__ volatile("dsb\n\tisb" ::: "memory");
    __asm__ volatile("mov %0, sp" : "=r"(sp));
    for (p = &_sstack; p < (uint32_t *)(sp - 64); p++)  /* водяной знак: всё ниже живого кадра */
        *p = STACK_MARK;
    initialise_monitor_handles();
    rc = main();
    for (p = &_sstack; p < &_estack && *p == STACK_MARK; p++)
        ;
    printf("baremetal: стек, максимум глубины %lu B из %lu B\n",
           (unsigned long)((uintptr_t)&_estack - (uintptr_t)p),
           (unsigned long)((uintptr_t)&_estack - (uintptr_t)&_sstack));
    fflush(stdout);
    semihost_exit(rc != 0);
}
