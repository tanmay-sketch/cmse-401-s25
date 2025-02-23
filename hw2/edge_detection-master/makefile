CFLAGS = -Os
all: process 

main_process.o: main_process.c
	gcc $(CFLAGS) -c main_process.c 

png_util.o: png_util.c
	gcc $(CFLAGS) -l lpng16 -c png_util.c

process: main_process.o png_util.o
	gcc $(CFLAGS) -o process -lm -l png16 main_process.o png_util.o

test: process 
	./process ./images/cube.png test.png

clean:
	rm *.o
	rm process

 
