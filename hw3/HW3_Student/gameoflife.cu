#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "png_util.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define MAX_N 20000
#define CUDA_CALL(x) { cudaError_t cuda_error__ = (x); if (cuda_error__) std::cout << "CUDA error: " << #x << " returned " << cudaGetErrorString(cuda_error__) << std::endl; }

char plate[2][(MAX_N + 2) * (MAX_N + 2)];
int which = 0;
int n;

// CUDA kernel to perform one iteration of the Game of Life on the GPU.
// It takes a pointer to the GPU plate memory, the grid size n, and which board is current.
__global__ void iteration_kernel(char* d_plate, int n, int which) {
    // Calculate the row and column indices (the board is padded with an extra border)
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    // FIX: Use threadIdx.x (not threadIdx.y) to compute the column index.
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (i <= n && j <= n) {
        int stride = n + 2;
        int index = i * stride + j;
        int plate_size = (n + 2) * (n + 2);
        // Determine pointers to the current board and the next board.
        char* curr = d_plate + which * plate_size;
        char* next = d_plate + ((which ^ 1) * plate_size);
        // Count live neighbors.
        int num = curr[index - stride - 1] + curr[index - stride] + curr[index - stride + 1] +
                  curr[index - 1] + curr[index + 1] +
                  curr[index + stride - 1] + curr[index + stride] + curr[index + stride + 1];
        // Apply Game of Life rules.
        if (curr[index])
            next[index] = (num == 2 || num == 3) ? 1 : 0;
        else
            next[index] = (num == 3);
    }
}

// Print the CPU plate (only if n is small enough to display)
void print_plate(){
    if (n < 60) {
        for (int i = 1; i <= n; i++){
            for (int j = 1; j <= n; j++){
                printf("%d", (int) plate[which][i * (n + 2) + j]);
            }
            printf("\n");
        }
    } else {
        printf("Plate too large to print to screen\n");
    }
    printf("\0");
}

// Convert the CPU plate to a PNG image file.
// The filename parameter remains const, but we cast it to char* when calling write_png_file.
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
            if (plate[!which][pindex] > 0)
                img[index] = 255; 
            else 
                img[index] = 0;
        }
    }
    printf("Writing file\n");
    // Cast filename to char* to match write_png_file's parameter type.
    write_png_file((char*)filename, img, sz);
    printf("done writing png\n"); 
    free(img);
    printf("done freeing memory\n");
}

int main() {
    int M;
    char line[MAX_N];
    
    // ------- GPU initialization -------
    // Set device to GPU 1 which has plenty of available memory.
    CUDA_CALL(cudaSetDevice(1));
    
    if (scanf("%d %d", &n, &M) == 2) {
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
            n = MAX_N; 
            for (int i = 1; i <= n; i++) 
                for (int j = 0; j < n; j++) 
                    plate[0][i * (n + 2) + j + 1] = (char) (rand() % 2);
        }

        // Calculate GPU memory size for one board and allocate for two boards.
        int plate_size = (n + 2) * (n + 2) * sizeof(char);
        int total_size = 2 * plate_size;
        char* d_plate;

        // -------- Allocating and copying to GPU ---------
        CUDA_CALL(cudaMalloc((void**)&d_plate, total_size));
        CUDA_CALL(cudaMemcpy(d_plate, plate[0], plate_size, cudaMemcpyHostToDevice));
        
        // Define block and grid dimensions.
        dim3 block(16, 16); // defines a block of 16x16 threads
        dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y); // calculates number of blocks needed
        
        // Main simulation loop on the CPU side, invoking the GPU kernel each iteration.
        for (int i = 0; i < M; i++) {
            if (n < 60) {
                CUDA_CALL(cudaMemcpy(plate[which], d_plate + which * plate_size, plate_size, cudaMemcpyDeviceToHost));
                printf("\nIteration %d:\n", i);
                print_plate();
            }
            iteration_kernel<<<grid, block>>>(d_plate, n, which);
            CUDA_CALL(cudaGetLastError()); // checks for any kernel launch error
            CUDA_CALL(cudaDeviceSynchronize()); // ensures kernel execution is complete
            which ^= 1; // switches the board (using xor) so that the next iteration updates the correct board
        }
        CUDA_CALL(cudaMemcpy(plate[which], d_plate + which * plate_size, plate_size, cudaMemcpyDeviceToHost));
        plate2png("plate.png");
        print_plate();
        CUDA_CALL(cudaFree(d_plate));
    }
    return 0;
}
