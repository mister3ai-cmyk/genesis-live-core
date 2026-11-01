CC      ?= gcc
CFLAGS  ?= -O3 -march=native -std=c99 -Wall -Wextra -pthread
LDLIBS   = -lm
# librt is a separate library on glibc only; on macOS and musl the
# clock_gettime/shm_open symbols live in libc and -lrt fails to link.
ifeq ($(shell uname -s),Linux)
LDLIBS  += -lrt
endif
SRC      = ../src

run_benchmark: run_benchmark.c $(SRC)/srp_lsh.c $(SRC)/shm_ring.c $(SRC)/srp_lsh.h $(SRC)/shm_ring.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ run_benchmark.c $(SRC)/srp_lsh.c $(SRC)/shm_ring.c $(LDLIBS)

cold_cache: cold_cache.c $(SRC)/srp_lsh.c $(SRC)/shm_ring.c $(SRC)/srp_lsh.h $(SRC)/shm_ring.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ cold_cache.c $(SRC)/srp_lsh.c $(SRC)/shm_ring.c $(LDLIBS)

clean:
	rm -f run_benchmark cold_cache *.o

.PHONY: clean
