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
// We store two boards in a contiguous block on the host.
// (The CPU version later uses plate[!which] when writing the PNG.)
char plate[2][(MAX_N + 2) * (MAX_N + 2)];
int n;

// CUDA kernel to perform one iteration of the Game of Life.
// The kernel reads from board "gpu_which" (located at d_plate offset gpu_which*plate_size)
// and writes to board (gpu_which ^ 1).
__global__ void iteration_kernel(char* d_plate, int n, int gpu_which) {
    // Compute row and column indices (accounting for the padded border)
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i <= n && j <= n) {
        int stride = n + 2;
        int index = i * stride + j;
        int plate_size = (n + 2) * (n + 2);
        // Set pointers to current and next state
        char* curr = d_plate + gpu_which * plate_size;
        char* next = d_plate + ((gpu_which ^ 1) * plate_size);
        // Count live neighbors
        int num = curr[index - stride - 1] + curr[index - stride] + curr[index - stride + 1] +
                  curr[index - 1]            +                      curr[index + 1] +
                  curr[index + stride - 1]   + curr[index + stride]   + curr[index + stride + 1];
        // Apply Game of Life rules
        if (curr[index])
            next[index] = (num == 2 || num == 3) ? 1 : 0;
        else
            next[index] = (num == 3);
    }
}

// This version of print_plate simply prints the host board 0 (for debugging).
void print_plate_cuda(){
    if (n < 60) {
        for (int i = 1; i <= n; i++){
            for (int j = 1; j <= n; j++){
                printf("%d", (int) plate[0][i*(n+2)+j]);
            }
            printf("\n");
        }
    } else {
        printf("Plate too large to print to screen\n");
    }
    printf("\0");
}

// In the CPU code, plate2png uses the board opposite to the final 'which' value.
// That is, if after M iterations the CPU's "which" is 0, then it writes plate[!0] (i.e. plate[1]).
// To match that exactly, we compute final_board = (gpu_which ^ 1) and use the same (1-indexed) output index.
void plate2png_cuda(const char* filename, int final_board) {
    unsigned char* img = (unsigned char*) malloc(n * n * sizeof(unsigned char));
    if (img == NULL) {
        fprintf(stderr, "Error allocating memory for image\n");
        exit(EXIT_FAILURE);
    }
    image_size_t sz;
    sz.width = n;
    sz.height = n; 
    // Mimic the CPU code's indexing: for i from 1 to n and j from 1 to n,
    // use pindex = i*(n+2) + j and index = (i-1)*n + j (note: CPU code uses 1-indexed output).
    for (int i = 1; i <= n; i++){
        for (int j = 1; j <= n; j++){
            int pindex = i * (n + 2) + j;
            int index = (i - 1) * n + j;  // Matches CPU code's (off-by-one) indexing.
            if (plate[final_board][pindex] > 0)
                img[index] = 255; 
            else 
                img[index] = 0;
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
    int gpu_which = 0;  // Initialize GPU toggling variable (matches CPU code's 'which')
    // Read grid size and number of iterations.
    if (scanf("%d %d", &n, &M) == 2) {
        if (n > 0) {
            // Initialize host board: board 0 holds the initial state; board 1 is zeroed.
            memset(plate[0], 0, sizeof(char) * (n + 2) * (n + 2));
            memset(plate[1], 0, sizeof(char) * (n + 2) * (n + 2));
            for (int i = 1; i <= n; i++){
                scanf("%s", line);
                for (int j = 0; j < n; j++){
                    plate[0][i * (n + 2) + j + 1] = line[j] - '0';
                }
            }
        } else {
            n = MAX_N; 
            for (int i = 1; i <= n; i++)
                for (int j = 0; j < n; j++)
                    plate[0][i * (n + 2) + j + 1] = (char)(rand() % 2);
        }
        
        // Calculate memory size for one board and allocate one contiguous block for two boards.
        int plate_size = (n + 2) * (n + 2) * sizeof(char);
        int total_size = 2 * plate_size;
        char* d_plate;
        CUDA_CALL(cudaMalloc((void**)&d_plate, total_size));
        // Copy initial state (board 0) from host to GPU.
        CUDA_CALL(cudaMemcpy(d_plate, plate[0], plate_size, cudaMemcpyHostToDevice));
        
        // Set up kernel launch dimensions.
        dim3 block(16, 16);
        dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
        
        // Main simulation loop: each iteration runs the kernel and toggles gpu_which.
        for (int i = 0; i < M; i++) {
            iteration_kernel<<<grid, block>>>(d_plate, n, gpu_which);
            CUDA_CALL(cudaGetLastError());
            CUDA_CALL(cudaDeviceSynchronize());
            gpu_which ^= 1;  // Toggle the board index.
        }
        // According to the CPU code, the final PNG is written from the board opposite to 'which'.
        // That is, if CPU's final which is 0, PNG is generated from plate[!0] (i.e. plate[1]),
        // and vice-versa. So, compute final_board = (gpu_which ^ 1).
        int final_board = gpu_which ^ 1;
        // Copy final state from GPU board 'final_board' to host board 'final_board'
        CUDA_CALL(cudaMemcpy(plate[final_board], d_plate + final_board * plate_size, plate_size, cudaMemcpyDeviceToHost));
        // Generate PNG using the same board indexing as the CPU code.
        plate2png_cuda("plate_parallel.png", final_board);
        print_plate_cuda();
        CUDA_CALL(cudaFree(d_plate));
    }
    return 0;
}