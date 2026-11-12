# Business Source License 1.1 - см. research/LICENSE
# rt_core: C99, без malloc/free; встраиваемые цели - без libc и libm.
CC      ?= gcc
OPT     ?= -O2
WARN     = -std=c99 -Wall -Wextra -Werror -pedantic
CFLAGS   = $(OPT) $(WARN) -fPIC
LDLIBS   = -lm
SRCS     = rt_kinematics.c rt_grassmann.c rt_xcorr.c
OBJS     = $(SRCS:.c=.o)
HDRS     = rt_core.h rt_port.h rt_kinematics.h rt_grassmann.h rt_xcorr.h

# встраиваемая сборка: без ОС, без защиты стека и FORTIFY (их символов на
# голом железе нет), циклы не превращать обратно в memset/memcpy
EMB      = -O2 $(WARN) -ffreestanding -fno-math-errno -fno-stack-protector \
           -U_FORTIFY_SOURCE -fno-tree-loop-distribute-patterns -fstack-usage
A64_CC  ?= aarch64-linux-gnu-gcc
ARM_CC  ?= arm-none-eabi-gcc
QEMU    ?= qemu-aarch64
M4       = -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 -DRT_NO_LIBM
M7       = -mcpu=cortex-m7 -mthumb -mfloat-abi=hard -mfpu=fpv5-d16
# допустимые внешние символы встраиваемых объектов: служебные функции
# libgcc (программная двойная точность на M4: __aeabi_d*, деление int64)
EMB_OK   = ^(__aeabi_|__(u?div|u?mod)di3$$)

all: librt_core.so tests/test_rt bench/bench_rt

%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) -c -o $@ $<

librt_core.so: $(OBJS)
	$(CC) -shared -o $@ $(OBJS) $(LDLIBS)

tests/test_rt: tests/test_rt.c $(OBJS)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_rt.c $(OBJS) $(LDLIBS)

bench/bench_rt: bench/bench_rt.c $(OBJS)
	$(CC) $(CFLAGS) -I. -DRT_BENCH_FLAGS='"$(OPT) $(WARN)"' -o $@ bench/bench_rt.c $(OBJS) $(LDLIBS)

check: tests/test_rt
	./tests/test_rt

# запрет динамической памяти: ни одной ссылки на malloc/calloc/realloc/free
check-nomalloc: $(OBJS)
	@if nm -u $(OBJS) | grep -Ew '(malloc|calloc|realloc|free)'; then \
		echo "FAIL: динамическая память в rt_core"; exit 1; \
	else echo "check-nomalloc: ok ($(OBJS))"; fi

bench: bench/bench_rt
	./bench/bench_rt

# ARM64 (Linux): ядро без libc и libm, тесты статически, прогон под qemu
cross-aarch64:
	@rm -rf build/a64 *.su; mkdir -p build/a64
	@for f in $(SRCS); do $(A64_CC) $(EMB) -c -o build/a64/$${f%.c}.o $$f || exit 1; done
	@mv -f *.su build/a64/ 2>/dev/null || true
	@bad=$$(aarch64-linux-gnu-nm -u build/a64/*.o | grep -vE '^$$|:$$' | awk '{print $$2}' | grep -vE '$(EMB_OK)'); \
	if [ -n "$$bad" ]; then echo "FAIL aarch64: внешние символы: $$bad"; exit 1; fi
	$(A64_CC) -O2 $(WARN) -static -I. -o build/a64/test_rt tests/test_rt.c build/a64/*.o -lm
	$(QEMU) build/a64/test_rt
	@echo "cross-aarch64: ok, внешних символов в ядре нет"

# Cortex-M4F (FPU одинарной точности: double программно) и M7 (FPU двойной)
cross-m4 cross-m7:
	@rm -rf build/$(@:cross-%=%) *.su; mkdir -p build/$(@:cross-%=%)
	@for f in $(SRCS); do $(ARM_CC) $(EMB) $(if $(filter cross-m4,$@),$(M4),$(M7)) -c \
		-o build/$(@:cross-%=%)/$${f%.c}.o $$f || exit 1; done
	@mv -f *.su build/$(@:cross-%=%)/ 2>/dev/null || true
	@bad=$$(arm-none-eabi-nm -u build/$(@:cross-%=%)/*.o | grep -vE '^$$|:$$' | awk '{print $$2}' | grep -vE '$(EMB_OK)' | sort -u); \
	if [ -n "$$bad" ]; then echo "FAIL $@: внешние символы: $$bad"; exit 1; fi
	@echo "$@: внешние символы (libgcc): $$(arm-none-eabi-nm -u build/$(@:cross-%=%)/*.o | awk 'NF==2{print $$2}' | sort -u | tr '\n' ' ')"
	@arm-none-eabi-size -t build/$(@:cross-%=%)/*.o
	@echo "стек по функциям, байт (верх):"; cat build/$(@:cross-%=%)/*.su | sort -t'	' -k2 -n -r | head -6

embedded: cross-aarch64 cross-m4 cross-m7

clean:
	rm -rf $(OBJS) librt_core.so tests/test_rt bench/bench_rt build *.su

.PHONY: all check check-nomalloc bench clean cross-aarch64 cross-m4 cross-m7 embedded
