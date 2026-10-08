CC      ?= gcc
CFLAGS  ?= -O3 -march=native -std=c99 -Wall -Wextra -pthread
LDLIBS   = -lrt -lm
SRC      = ../src

run_benchmark: run_benchmark.c $(SRC)/srp_lsh.c $(SRC)/shm_ring.c $(SRC)/srp_lsh.h $(SRC)/shm_ring.h
	$(CC) $(CFLAGS) -o $@ run_benchmark.c $(SRC)/srp_lsh.c $(SRC)/shm_ring.c $(LDLIBS)

clean:
	rm -f run_benchmark *.o

.PHONY: clean
