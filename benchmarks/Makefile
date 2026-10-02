CC       = gcc
CFLAGS   = -O3 -march=native -std=c99 -Wall -Wextra -pedantic
LDFLAGS  = -lpthread -lrt
SHFLAGS  = -shared -fPIC

TARGETS  = c_shm_harness fast_hamming.so

all: $(TARGETS)

c_shm_harness: c_shm_harness.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

fast_hamming.so: fast_hamming.c
	$(CC) $(CFLAGS) $(SHFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean
