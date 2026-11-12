# Business Source License 1.1 - см. research/LICENSE
# rt_core: C99, только libc и libm, без malloc/free.
CC      ?= gcc
OPT     ?= -O2
WARN     = -std=c99 -Wall -Wextra -Werror -pedantic
CFLAGS   = $(OPT) $(WARN) -fPIC
LDLIBS   = -lm
OBJS     = rt_kinematics.o rt_grassmann.o
HDRS     = rt_core.h rt_kinematics.h rt_grassmann.h

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

clean:
	rm -f $(OBJS) librt_core.so tests/test_rt bench/bench_rt

.PHONY: all check check-nomalloc bench clean
