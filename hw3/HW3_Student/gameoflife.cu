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

        // Count live neighbors (same offsets as CPU code).
        int num = curr[index - stride - 1] + curr[index - stride] + curr[index - stride + 1]
                + curr[index - 1]         +                          curr[index + 1]
                + curr[index + stride - 1] + curr[index + stride]    + curr[index + stride + 1];

        // Apply Game of Life rules.
        if (curr[index])
            next[index] = (num == 2 || num == 3) ? 1 : 0;
        else
            next[index] = (num == 3);
    }
}

// -------------------------------------------------------------------
// For debugging: prints plate[0] if n < 60. (Optional)
void print_plate_cuda() {
    if (n < 60) {
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                printf("%d", (int) plate[0][i * (n + 2) + j]);
            }
            printf("\n");
        }
    } else {
        printf("Plate too large to print to screen\n");
    }
    printf("\0");
}

// -------------------------------------------------------------------
// This PNG function matches the CPU code’s indexing, which uses:
//   - (i-1)*n + j  for the 1D image index
//   - pindex = i*(n+2) + j for reading from board
//   - final board is plate[!which] in the CPU code
// So we replicate that here by taking a final_board argument.
void plate2png_cuda(const char* filename, int final_board) {
    unsigned char* img = (unsigned char*) malloc(n * n * sizeof(unsigned char));
    if (img == NULL) {
        fprintf(stderr, "Error allocating memory for image\n");
        exit(EXIT_FAILURE);
    }
    image_size_t sz;
    sz.width = n;
    sz.height = n; 

    // The CPU code does:
    //   int index = (i-1)*n + j;
    //   int pindex = i*(n+2) + j;
    // for i, j in [1..n].
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            int pindex = i * (n + 2) + j;
            int index  = (i - 1) * n + j;  // Off-by-one style used by the CPU code
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

    // Use GPU 1 (or whichever device is best on your system).
    CUDA_CALL(cudaSetDevice(1));

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
        } else {
            // If n <= 0, set n = MAX_N and fill randomly
            n = MAX_N;
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
        
        // Zero out the entire buffer to avoid any leftover data (gray squares).
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

        // The CPU code writes its final PNG from plate[!which].
        // So if gpu_which is 0, the final data is in board1. If gpu_which is 1, final data is in board0.
        // final_board = gpu_which ^ 1 ensures we do the same as CPU code.
        int final_board = gpu_which ^ 1;

        // Copy that final board from the GPU to the host
        CUDA_CALL(cudaMemcpy(plate[final_board],
                             d_plate + final_board * plate_size,
                             plate_size,
                             cudaMemcpyDeviceToHost));

        // Write PNG from board "final_board" using the CPU code's indexing
        plate2png_cuda("plate_parallel.png", final_board);

        // (Optional) Print a small board for debugging. This always prints plate[0].
        // If you want to see the final board exactly, you might do:
        //    if (final_board == 1) memcpy(plate[0], plate[1], plate_size);
        //    print_plate_cuda();

        // Clean up
        CUDA_CALL(cudaFree(d_plate));
    }
    return 0;
}