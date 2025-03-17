#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "png_util.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define MAX_N 20000
#define CUDA_CALL(x) { \
    cudaError_t cuda_error__ = (x); \
    if (cuda_error__) \
        std::cout << "CUDA error: " << #x << " returned " \
                  << cudaGetErrorString(cuda_error__) << std::endl; \
}

// We keep a global CPU array "plate" just for input/output.
// The final GPU result is copied back into plate[final_board].
char plate[2][(MAX_N + 2) * (MAX_N + 2)];
int n;

// -------------------------------------------------------------------
// CUDA kernel: reads from GPU board "gpu_which" and writes to "gpu_which ^ 1".
__global__ void iteration_kernel(char* d_plate, int n, int gpu_which) {
    // Compute row and column within [1..n] (we use a padded border).
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i <= n && j <= n) {
        int stride = n + 2;
        int index = i * stride + j;
        int plate_size = (n + 2) * (n + 2);

        // Current board is offset by gpu_which*plate_size.
        char* curr = d_plate + gpu_which * plate_size;
        // Next board is offset by (gpu_which ^ 1)*plate_size.
        char* next = d_plate + ((gpu_which ^ 1) * plate_size);

        // Count live neighbors exactly like the CPU implementation
        // The CPU version uses: index - n - 3, index - n - 2, index - n - 1, ...
        int num = curr[index - stride - 1] + curr[index - stride] + curr[index - stride + 1]
                + curr[index - 1]         +                          curr[index + 1]
                + curr[index + stride - 1] + curr[index + stride]    + curr[index + stride + 1];

        // Apply Game of Life rules.
        if (curr[index]) {
            next[index] = (num == 2 || num == 3) ? 1 : 0;
        } else {
            next[index] = (num == 3) ? 1 : 0;
        }
    }
}

// -------------------------------------------------------------------
// For debugging: prints plate[final_board] if n < 60.
void print_plate_cuda(int final_board) {
    if (n < 60) {
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                printf("%d", (int) plate[final_board][i * (n + 2) + j]);
            }
            printf("\n");
        }
    } else {
        printf("Plate too large to print to screen\n");
    }
}

// -------------------------------------------------------------------
// This PNG function matches the CPU code's indexing
void plate2png_cuda(const char* filename, int final_board) {
    unsigned char* img = (unsigned char*) malloc(n * n * sizeof(unsigned char));
    if (img == NULL) {
        fprintf(stderr, "Error allocating memory for image\n");
        exit(EXIT_FAILURE);
    }
    memset(img, 0, n * n * sizeof(unsigned char)); // Initialize to zeros
    
    image_size_t sz;
    sz.width = n;
    sz.height = n; 

    // Fill the image array - match exactly with CPU implementation
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            int pindex = i * (n + 2) + j;
            int index = (i - 1) * n + j - 1; // Shift j by 1 to get 0-based indexing
            
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

// -------------------------------------------------------------------
int main() {
    int M;
    char line[MAX_N];

    // Use GPU 0 (more commonly available)
    CUDA_CALL(cudaSetDevice(0));

    // Read n, M from stdin
    if (scanf("%d %d", &n, &M) == 2) {
        // Initialize the host board
        if (n > 0) {
            memset(plate[0], 0, sizeof(char) * (n + 2) * (n + 2));
            memset(plate[1], 0, sizeof(char) * (n + 2) * (n + 2));
            for (int i = 1; i <= n; i++){
                scanf("%s", line);
                for (int j = 0; j < n; j++){
                    plate[0][i * (n + 2) + j + 1] = line[j] - '0';
                }
            }
            
            // Print initial board
            printf("Initial board:\n");
            print_plate_cuda(0);
        } else {
            // If n <= 0, set n = MAX_N and fill randomly
            n = MAX_N;
            memset(plate[0], 0, sizeof(char) * (n + 2) * (n + 2));
            memset(plate[1], 0, sizeof(char) * (n + 2) * (n + 2));
            for (int i = 1; i <= n; i++) {
                for (int j = 0; j < n; j++) {
                    plate[0][i * (n + 2) + j + 1] = (char)(rand() % 2);
                }
            }
        }

        // Allocate a single GPU buffer for two boards.
        int plate_size = (n + 2) * (n + 2) * sizeof(char);
        int total_size = 2 * plate_size;
        char* d_plate = NULL;
        CUDA_CALL(cudaMalloc((void**)&d_plate, total_size));
        
        // Initialize both boards with zeros to avoid any leftover data
        CUDA_CALL(cudaMemset(d_plate, 0, total_size));

        // Copy the initial state (board0) to GPU offset 0
        CUDA_CALL(cudaMemcpy(d_plate, plate[0], plate_size, cudaMemcpyHostToDevice));

        // Set up block and grid dimensions
        dim3 block(16, 16);
        dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

        // Match CPU toggling: "which" is the CPU side, "gpu_which" is the GPU side
        int gpu_which = 0;

        // Run M iterations, toggling after each
        for (int i = 0; i < M; i++) {
            iteration_kernel<<<grid, block>>>(d_plate, n, gpu_which);
            CUDA_CALL(cudaGetLastError());
            CUDA_CALL(cudaDeviceSynchronize());
            gpu_which ^= 1;  // flip boards
        }

        // The final board is the opposite of the current gpu_which
        int final_board = gpu_which ^ 1;

        // Copy that final board from the GPU to the host
        CUDA_CALL(cudaMemcpy(plate[final_board],
                             d_plate + final_board * plate_size,
                             plate_size,
                             cudaMemcpyDeviceToHost));

        // Print the final board for debugging
        printf("\n\nFinal:\n");
        print_plate_cuda(final_board);

        // Write PNG from board "final_board" using the CPU code's indexing
        plate2png_cuda("plate_parallel.png", final_board);

        // Clean up
        CUDA_CALL(cudaFree(d_plate));
    }
    return 0;
}