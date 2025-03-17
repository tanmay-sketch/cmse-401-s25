all: gol

CC := nvcc

ifeq ($(CC),nvcc)
	EXT := cu
	CFLAGS := -std=c++11
	LDFLAGS := -lpng
else
	EXT := c
	CFLAGS := -O3
	LDFLAGS := -lm -lpng
endif

gameoflife.o: gameoflife.$(EXT)
	$(CC) $(CFLAGS) -c gameoflife.$(EXT)

png_util.o: png_util.$(EXT)
	$(CC) $(CFLAGS) -c png_util.$(EXT)

gol: gameoflife.o png_util.o
	$(CC) $(CFLAGS) -o gol gameoflife.o png_util.o $(LDFLAGS)

test: gol data.txt
	./gol < data.txt

test2: gol data.txt
	cuda-memcheck ./gol < data.txt

clean:
	rm -f *.o gol