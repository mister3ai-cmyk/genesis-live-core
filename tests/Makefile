CC      ?= gcc
CFLAGS  ?= -O2 -std=c99 -Wall -Wextra -Wno-unused-parameter
LDLIBS   = -lm
SRC      = ../src

test_srp_lsh: test_srp_lsh.c $(SRC)/srp_lsh.c $(SRC)/srp_lsh.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(SRC) -o $@ test_srp_lsh.c $(SRC)/srp_lsh.c $(LDLIBS)

check: test_srp_lsh
	./test_srp_lsh

clean:
	rm -f test_srp_lsh *.o

.PHONY: check clean
