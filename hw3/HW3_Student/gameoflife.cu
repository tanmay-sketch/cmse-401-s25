#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "png_util.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define MAX_N 20000
#define CUDA_CALL(x) { cudaError_t cuda_error__ = (x); \
    if (cuda_error__) std::cout << "CUDA error: " << #x << " returned " \
    << cudaGetErrorString(cuda_error__) << std::endl; }

// CPU simulation boards (for input/output).
// We only use plate[0] for I/O, since the final state is stored there.
char plate[2][(MAX_N + 2) * (MAX_N + 2)];
int n;

// CUDA kernel to perform one iteration of the Game of Life.
// This kernel always reads from board0 (at d_plate offset 0)
// and writes to board1 (at d_plate + plate_size).
__global__ void iteration_kernel(char* d_plate, int n) {
    // Compute row and column indices (accounting for the padded border)
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    
    if (i <= n && j <= n) {
        int stride = n + 2;
        int index = i * stride + j;
        int plate_size = (n + 2) * (n + 2);
        // Pointers to current and next state
        char* curr = d_plate;                // board0: current state
        char* next = d_plate + plate_size;     // board1: next state
        
        // Count live neighbors
        int num = curr[index - stride - 1] + curr[index - stride] + curr[index - stride + 1] +
                  curr[index - 1]           +                      curr[index + 1] +
                  curr[index + stride - 1]  + curr[index + stride]  + curr[index + stride + 1];
                  
        // Apply Game of Life rules
        if (curr[index])
            next[index] = (num == 2 || num == 3) ? 1 : 0;
        else
            next[index] = (num == 3);
    }
}

// Print the CPU board (only if n is small enough to display)
void print_plate(){
    if (n < 60) {
        for (int i = 1; i <= n; i++){
            for (int j = 1; j <= n; j++){
                printf("%d", (int) plate[0][i * (n + 2) + j]);
            }
            printf("\n");
        }
    } else {
        printf("Plate too large to print to screen\n");
    }
    printf("\0");
}

// Convert the CPU board to a PNG image file.
// The final simulation state is stored in plate[0].
void plate2png(const char* filename) {
    unsigned char* img = (unsigned char*) malloc(n * n * sizeof(unsigned char));
    if (img == NULL) {
        fprintf(stderr, "Error allocating memory for image\n");
        exit(EXIT_FAILURE);
    }
    image_size_t sz;
    sz.width = n;
    sz.height = n; 

    for (int i = 1; i <= n; i++){
        for (int j = 1; j <= n; j++){
            int pindex = i * (n + 2) + j;
            int index = (i - 1) * n + (j - 1);
            img[index] = (plate[0][pindex] > 0) ? 255 : 0;
        }
    }
    printf("Writing file\n");
    write_png_file((char*)filename, img, sz);
    printf("done writing png\n"); 
    free(img);
    printf("done freeing memory\n");
}

int main() {
    int M;
    char line[MAX_N];
    
    // Set GPU device to GPU 1 (which has plenty of free memory)
    CUDA_CALL(cudaSetDevice(1));
    
    // Read grid size and number of iterations.
    if (scanf("%d %d", &n, &M) == 2) {
        if (n > 0) {
            // Initialize CPU board (board0 holds the input state; board1 is unused)
            memset(plate[0], 0, sizeof(char) * (n + 2) * (n + 2));
            memset(plate[1], 0, sizeof(char) * (n + 2) * (n + 2));
            for (int i = 1; i <= n; i++){
                scanf("%s", line);
                for (int j = 0; j < n; j++){
                    plate[0][i * (n + 2) + j + 1] = line[j] - '0';
                }
            }
        } else {
            // If n <= 0, use MAX_N and initialize randomly.
            n = MAX_N; 
            for (int i = 1; i <= n; i++)
                for (int j = 0; j < n; j++)
                    plate[0][i * (n + 2) + j + 1] = (char)(rand() % 2);
        }
        
        // Calculate memory size for one board and allocate a contiguous block for two boards.
        int plate_size = (n + 2) * (n + 2) * sizeof(char);
        int total_size = 2 * plate_size;
        char* d_plate;
        
        // Allocate GPU memory.
        CUDA_CALL(cudaMalloc((void**)&d_plate, total_size));
        // Copy initial state (board0) from CPU to GPU.
        CUDA_CALL(cudaMemcpy(d_plate, plate[0], plate_size, cudaMemcpyHostToDevice));
        
        // Set up CUDA kernel launch dimensions.
        dim3 block(16, 16);
        dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
        
        // Main simulation loop:
        // For each iteration, run the kernel to compute the next state (board1),
        // then copy board1 back to board0 so that the state is updated.
        for (int i = 0; i < M; i++) {
            iteration_kernel<<<grid, block>>>(d_plate, n);
            CUDA_CALL(cudaGetLastError());
            CUDA_CALL(cudaDeviceSynchronize());
            CUDA_CALL(cudaMemcpy(d_plate, d_plate + plate_size, plate_size, cudaMemcpyDeviceToDevice));
        }
        // Copy the final state (board0) from GPU to CPU.
        CUDA_CALL(cudaMemcpy(plate[0], d_plate, plate_size, cudaMemcpyDeviceToHost));
        plate2png("plate_parallel.png");
        print_plate();
        CUDA_CALL(cudaFree(d_plate));
    }
    return 0;
}
