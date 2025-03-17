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
char plate[2][(MAX_N + 2) * (MAX_N + 2)];
int n;

// CUDA kernel: reads from GPU board "gpu_which" and writes to "gpu_which ^ 1".
__global__ void iteration_kernel(char* d_plate, int n, int gpu_which) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i <= n && j <= n) {
        int stride = n + 2;
        int index = i * stride + j;
        int plate_size = (n + 2) * (n + 2);

        char* curr = d_plate + gpu_which * plate_size;
        char* next = d_plate + ((gpu_which ^ 1) * plate_size);

        int num = curr[index - stride - 1] + curr[index - stride] + curr[index - stride + 1]
                + curr[index - 1]         +                          curr[index + 1]
                + curr[index + stride - 1] + curr[index + stride]    + curr[index + stride + 1];

        if (curr[index]) {
            next[index] = (num == 2 || num == 3) ? 1 : 0;
        } else {
            next[index] = (num == 3) ? 1 : 0;
        }
    }
}

// For debugging: prints plate[board] if n < 60.
void print_plate_cuda(int board) {
    if (n < 60) {
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                printf("%d", (int) plate[board][i * (n + 2) + j]);
            }
            printf("\n");
        }
    } else {
        printf("Plate too large to print to screen\n");
    }
    printf("\0");
}

// This PNG function matches the CPU code's indexing
void plate2png_cuda(const char* filename, int final_board) {
    char* img = (char*) malloc(n * n * sizeof(char));
    image_size_t sz;
    sz.width = n;
    sz.height = n; 

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            int pindex = i * (n + 2) + j;
            int index = (i - 1) * n + j;
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

    // Try device 1 first, fall back to 0 if needed
    if (cudaSetDevice(1) != cudaSuccess) {
        CUDA_CALL(cudaSetDevice(0));
    }

    if (scanf("%d %d", &n, &M) == 2) {
        if (n > 0) {
            memset(plate[0], 0, sizeof(char) * (n + 2) * (n + 2));
            memset(plate[1], 0, sizeof(char) * (n + 2) * (n + 2));
            for (int i = 1; i <= n; i++) {
                scanf("%s", line);
                for (int j = 0; j < n; j++) {
                    plate[0][i * (n + 2) + j + 1] = line[j] - '0';
                }
            }
        } else {
            n = MAX_N;
            memset(plate[0], 0, sizeof(char) * (n + 2) * (n + 2));
            memset(plate[1], 0, sizeof(char) * (n + 2) * (n + 2));
            for (int i = 1; i <= n; i++) {
                for (int j = 0; j < n; j++) {
                    plate[0][i * (n + 2) + j + 1] = (char)(rand() % 2);
                }
            }
        }

        int plate_size = (n + 2) * (n + 2) * sizeof(char);
        int total_size = 2 * plate_size;
        char* d_plate = NULL;

        // Attempt to allocate memory with error handling
        cudaError_t malloc_result = cudaMalloc((void**)&d_plate, total_size);
        if (malloc_result != cudaSuccess) {
            fprintf(stderr, "CUDA memory allocation failed: %s\n", 
                    cudaGetErrorString(malloc_result));
            // Fall back to CPU implementation if GPU memory allocation fails
            fprintf(stderr, "Falling back to CPU implementation\n");
            
            // Copy CPU implementation here
            int which = 0;
            for (int i = 0; i < M; i++) {
                printf("\nIteration %d:\n", i);
                print_plate_cuda(which);
                
                // Simplified OpenMP-like iteration
                for (int i = 1; i <= n; i++) {
                    for (int j = 1; j <= n; j++) {
                        int index = i * (n + 2) + j;
                        int stride = n + 2;
                        // Count neighbors
                        int num = plate[which][index - stride - 1] + plate[which][index - stride] + 
                                  plate[which][index - stride + 1] + plate[which][index - 1] + 
                                  plate[which][index + 1] + plate[which][index + stride - 1] + 
                                  plate[which][index + stride] + plate[which][index + stride + 1];
                        
                        if (plate[which][index]) {
                            plate[!which][index] = (num == 2 || num == 3) ? 1 : 0;
                        } else {
                            plate[!which][index] = (num == 3) ? 1 : 0;
                        }
                    }
                }
                which = !which;
            }
            
            printf("\n\nFinal:\n");
            plate2png_cuda("plate_parallel.png", !which);
            print_plate_cuda(which);
            
            return 0;
        }

        // Initialize both boards with zeros
        CUDA_CALL(cudaMemset(d_plate, 0, total_size));
        CUDA_CALL(cudaMemcpy(d_plate, plate[0], plate_size, cudaMemcpyHostToDevice));

        dim3 block(16, 16);
        dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

        int gpu_which = 0;

        for (int i = 0; i < M; i++) {
            printf("\nIteration %d:\n", i);
            print_plate_cuda(i > 0 ? (gpu_which ^ 1) : 0);
            
            iteration_kernel<<<grid, block>>>(d_plate, n, gpu_which);
            cudaError_t kernel_error = cudaGetLastError();
            if (kernel_error != cudaSuccess) {
                fprintf(stderr, "Kernel execution failed: %s\n", 
                        cudaGetErrorString(kernel_error));
                cudaFree(d_plate);
                return 1;
            }
            
            CUDA_CALL(cudaDeviceSynchronize());
            gpu_which ^= 1;
        }

        int final_board = gpu_which ^ 1;
        CUDA_CALL(cudaMemcpy(plate[final_board],
                          d_plate + final_board * plate_size,
                          plate_size,
                          cudaMemcpyDeviceToHost));

        printf("\n\nFinal:\n");
        plate2png_cuda("plate_parallel.png", final_board);
        print_plate_cuda(final_board);

        CUDA_CALL(cudaFree(d_plate));
    }
    return 0;
}