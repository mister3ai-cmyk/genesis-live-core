CC       ?= gcc
CPPFLAGS ?= -DSRP_BITS=256
CFLAGS   ?= -O3 -march=native -std=c99 -Wall -Wextra -fPIC
LDLIBS    = -lm
SRC       = ../../src

libgenesis_l0.so: l0_api.c $(SRC)/srp_lsh.c $(SRC)/srp_lsh.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(SRC) -shared -o $@ l0_api.c $(SRC)/srp_lsh.c $(LDLIBS)

clean:
	rm -f libgenesis_l0.so

.PHONY: clean
