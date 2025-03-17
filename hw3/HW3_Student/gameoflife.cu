#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <stdlib.h>
#include <assert.h>
#include "png_util.h"
#include <cuda_runtime.h>

#define MAX_N 2000
#define CUDA_CALL(x) do { cudaError_t err = (x); if (err != cudaSuccess) { printf("Error %d: %s at %s:%d\n", err, cudaGetErrorString(err), __FILE__, __LINE__); exit(EXIT_FAILURE); } } while(0)

char plate[2][(MAX_N + 2) * (MAX_N + 2)];
int which = 0;
int n;

// CUDA kernel to update the plate state
__global__ void update_plate_kernel(char* d_plate_in, char* d_plate_out, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= 1 && row <= n && col >= 1 && col <= n) {
        int index = row * (n + 2) + col;
        
        // Count live neighbors (same as the CPU 'live' function)
        int num = d_plate_in[index - n - 3] + 
                  d_plate_in[index - n - 2] + 
                  d_plate_in[index - n - 1] + 
                  d_plate_in[index - 1] + 
                  d_plate_in[index + 1] + 
                  d_plate_in[index + n + 1] + 
                  d_plate_in[index + n + 2] + 
                  d_plate_in[index + n + 3];
        
        // Apply Game of Life rules
        if (d_plate_in[index]) {
            d_plate_out[index] = (num == 2 || num == 3) ? 1 : 0;
        } else {
            d_plate_out[index] = (num == 3) ? 1 : 0;
        }
    }
}

void cuda_iteration(char* d_plate_0, char* d_plate_1, int n, int* which) {
    // Define grid and block dimensions - smaller blocks to avoid using too many resources
    dim3 blockDim(8, 8);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);
    
    // Print kernel launch configuration
    printf("Launching kernel with grid: (%d, %d), block: (%d, %d)\n", 
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    
    // Launch kernel - alternating input and output buffers based on 'which'
    if (*which == 0) {
        update_plate_kernel<<<gridDim, blockDim>>>(d_plate_0, d_plate_1, n);
    } else {
        update_plate_kernel<<<gridDim, blockDim>>>(d_plate_1, d_plate_0, n);
    }
    
    // Check for kernel launch errors
    CUDA_CALL(cudaGetLastError());
    
    // Synchronize to ensure kernel execution is complete
    CUDA_CALL(cudaDeviceSynchronize());
    
    // Flip the buffer selector
    *which = !(*which);
}

void print_plate() {
    if (n < 60) {
        for(int i = 1; i <= n; i++) {
            for(int j = 1; j <= n; j++) {
                printf("%d", (int) plate[which][i * (n + 2) + j]);
            }
            printf("\n");
        }
    } else {
        printf("Plate too large to print to screen\n");
    }
}

void plate2png(char* filename) {
    unsigned char* img = (unsigned char*) malloc(n*n*sizeof(unsigned char));
    image_size_t sz;
    sz.width = n;
    sz.height = n; 
    
    for(int i = 1; i <= n; i++) {
        for(int j = 1; j <= n; j++) {
            int pindex = i * (n + 2) + j;
            int index = (i-1) * n + (j-1);
            if (plate[which][pindex] > 0)
                img[index] = 255; 
            else 
                img[index] = 0;
        }
    }
    
    printf("Writing file\n");
    write_png_file(filename, img, sz);
    
    printf("done writing png\n"); 
    free(img);
    printf("done freeing memory\n");
}

int main() { 
    int M;
    char line[MAX_N];
    
    if(scanf("%d %d", &n, &M) == 2) {
        // If n <= 0, set it to a more reasonable size for random generation
        if (n <= 0) {
            // Use a smaller value than MAX_N to avoid GPU memory issues
            n = 1000; // Reduced from MAX_N (20000) to fit in GPU memory
            printf("Using n = %d for random generation\n", n);
        }

        // Initialize plates
        memset(plate[0], 0, sizeof(char) * (n + 2) * (n + 2));
        memset(plate[1], 0, sizeof(char) * (n + 2) * (n + 2));
        
        // Fill the initial state
        if (n == 1000) { // This is our random case
            for(int i = 1; i <= n; i++) 
                for(int j = 0; j < n; j++) 
                    plate[0][i * (n+2) + j + 1] = (char) rand() % 2;
        } else {
            for(int i = 1; i <= n; i++) {
                scanf("%s", line);
                for(int j = 0; j < n; j++) {
                    plate[0][i * (n + 2) + j + 1] = line[j] - '0';
                }
            }
        }
        
        // Allocate device memory
        char *d_plate_0 = NULL;
        char *d_plate_1 = NULL;
        size_t plate_size = sizeof(char) * (n + 2) * (n + 2);
        
        printf("Allocating %zu bytes of GPU memory\n", 2 * plate_size);
        
        CUDA_CALL(cudaMalloc((void**)&d_plate_0, plate_size));
        CUDA_CALL(cudaMalloc((void**)&d_plate_1, plate_size));
        
        // Copy initial state to device
        CUDA_CALL(cudaMemcpy(d_plate_0, plate[0], plate_size, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_plate_1, plate[1], plate_size, cudaMemcpyHostToDevice));
        
        // Main simulation loop
        int gpu_which = 0;  // Track which buffer is current on GPU
        
        printf("\nIteration %d:\n", 0);
        print_plate();
        
        for(int i = 0; i < M; i++) {
            // Run iteration on GPU
            cuda_iteration(d_plate_0, d_plate_1, n, &gpu_which);
            
            if (i < M-1) {
                printf("\nIteration %d:\n", i+1);
                
                // Copy current state back to host for display (optional)
                if (gpu_which == 0) {
                    CUDA_CALL(cudaMemcpy(plate[0], d_plate_0, plate_size, cudaMemcpyDeviceToHost));
                    which = 0;
                } else {
                    CUDA_CALL(cudaMemcpy(plate[1], d_plate_1, plate_size, cudaMemcpyDeviceToHost));
                    which = 1;
                }
                
                print_plate();
            }
        }
        
        // Copy final state back to host
        if (gpu_which == 0) {
            CUDA_CALL(cudaMemcpy(plate[0], d_plate_0, plate_size, cudaMemcpyDeviceToHost));
            which = 0;
        } else {
            CUDA_CALL(cudaMemcpy(plate[1], d_plate_1, plate_size, cudaMemcpyDeviceToHost));
            which = 1;
        }
        
        printf("\n\nFinal:\n");
        char png_filename[] = "plate.png";
        plate2png(png_filename);
        print_plate();
        
        // Free device memory
        CUDA_CALL(cudaFree(d_plate_0));
        CUDA_CALL(cudaFree(d_plate_1));
    }
    
    return 0;
}
