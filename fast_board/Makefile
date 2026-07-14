CC ?= cc
CFLAGS ?= -O3 -march=native -fPIC -Wall -Wextra -Wno-unused-function
LDFLAGS ?= -shared

LIB = libboardgen.so
SRC = boardgen.c

.PHONY: all clean test

all: $(LIB)

$(LIB): $(SRC) boardgen.h
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(SRC)

clean:
	rm -f $(LIB)

test: $(LIB)
	python3 -c "from fast_board import _lib; print('perft1', _lib.perft(1)); print('perft2', _lib.perft(2)); print('perft3', _lib.perft(3)); print('perft4', _lib.perft(4)); print('perft5', _lib.perft(5))"
