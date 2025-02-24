CFLAGS_ORIGINAL = -O0
CFLAGS_BEST = -O3
CFLAGS_PARALLEL = -O3 -fopenmp
LIBS = -lm -lpng16

all: original best_serial process

original: original.c png_util.c
	gcc $(CFLAGS_ORIGINAL) -o original original.c png_util.c $(LIBS)

best_serial: best_serial_main.c png_util.c
	gcc $(CFLAGS_BEST) -o best_serial best_serial_main.c png_util.c $(LIBS)

process: main_process.c png_util.c
	gcc $(CFLAGS_PARALLEL) -o process main_process.c png_util.c $(LIBS)

test: process
	./process ./images/cube.png cube_edges.png
	./process ./images/earth.png earth_edges.png
	./process ./images/minion_bw.png minion_edges.png
	./process ./images/MSUStadium.png stadium_edges.png
	./process ./images/sparty.png sparty_edges.png

clean:
	rm -f original best_serial process *.o