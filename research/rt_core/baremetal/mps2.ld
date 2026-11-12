/* Business Source License 1.1 - см. research/LICENSE
 * Карта памяти qemu mps2-an386 (Cortex-M4) и mps2-an500 (Cortex-M7):
 * код с 0x00000000 (4 МБ), данные с 0x20000000 (4 МБ). Стек 64 КБ - в
 * конце RAM, куча newlib - между .bss и стеком. */
ENTRY(Reset_Handler)

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 4M
    RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 4M
}

STACK_SIZE = 64K;

SECTIONS
{
    .isr_vector : { KEEP(*(.isr_vector)) } > FLASH

    .text : {
        *(.text*) *(.rodata*)
        KEEP(*(.init)) KEEP(*(.fini))
        . = ALIGN(4);
    } > FLASH

    .ARM.extab : { *(.ARM.extab* .gnu.linkonce.armextab.*) } > FLASH
    .ARM.exidx : {
        __exidx_start = .;
        *(.ARM.exidx* .gnu.linkonce.armexidx.*)
        __exidx_end = .;
    } > FLASH

    .preinit_array : { PROVIDE_HIDDEN(__preinit_array_start = .); KEEP(*(.preinit_array*)); PROVIDE_HIDDEN(__preinit_array_end = .); } > FLASH
    .init_array    : { PROVIDE_HIDDEN(__init_array_start = .); KEEP(*(.init_array*)); PROVIDE_HIDDEN(__init_array_end = .); } > FLASH
    .fini_array    : { PROVIDE_HIDDEN(__fini_array_start = .); KEEP(*(.fini_array*)); PROVIDE_HIDDEN(__fini_array_end = .); } > FLASH

    _sidata = LOADADDR(.data);
    .data : {
        . = ALIGN(4);
        _sdata = .;
        *(.data*)
        . = ALIGN(4);
        _edata = .;
    } > RAM AT > FLASH

    .bss (NOLOAD) : {
        . = ALIGN(4);
        _sbss = .;
        *(.bss*) *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > RAM

    . = ALIGN(8);
    end = .;
    _end = .;
    PROVIDE(__end__ = .);

    _estack = ORIGIN(RAM) + LENGTH(RAM);
    _sstack = _estack - STACK_SIZE;
    __HeapLimit = _sstack;
    ASSERT(end <= _sstack, "RAM: .data + .bss не помещаются под стек")
}
