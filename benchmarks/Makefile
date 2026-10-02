CC      = gcc
CFLAGS  = -O3 -march=native -std=c99 -Wall -Wextra -pedantic
LDFLAGS = -lpthread -lrt
TARGET  = c_shm_harness

all: $(TARGET)

$(TARGET): c_shm_harness.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
