#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <stdlib.h>
#include <assert.h>
#include "png_util.h"
#include <cuda_runtime.h>

#define MAX_N 2000
#define CUDA_CALL(x) do { cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        printf("Error %d: %s at %s:%d\n", err, cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Static CPU plate: two boards each sized for a maximum grid of (MAX_N+2) x (MAX_N+2)
char plate[2][(MAX_N + 2) * (MAX_N + 2)];
int which = 0;  // This index tracks which CPU board is current.
int n;

// CUDA kernel to update the plate state on the GPU.
// It receives two GPU memory pointers (input and output boards) and the simulation size.
__global__ void update_plate_kernel(char* d_plate_in, char* d_plate_out, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Only update interior cells (the board is padded with an extra border)
    if (row >= 1 && row <= n && col >= 1 && col <= n) {
        int index = row * (n + 2) + col;
        
        // Count live neighbors (using the padded board, so indexing shifts by 1)
        int num = d_plate_in[index - n - 3] + 
                  d_plate_in[index - n - 2] + 
                  d_plate_in[index - n - 1] + 
                  d_plate_in[index - 1]     + 
                  d_plate_in[index + 1]     + 
                  d_plate_in[index + n + 1] + 
                  d_plate_in[index + n + 2] + 
                  d_plate_in[index + n + 3];
        
        // Game of Life rules:
        // A live cell (value 1) survives if it has 2 or 3 neighbors.
        // A dead cell becomes live if it has exactly 3 neighbors.
        if (d_plate_in[index]) {
            d_plate_out[index] = (num == 2 || num == 3) ? 1 : 0;
        } else {
            d_plate_out[index] = (num == 3) ? 1 : 0;
        }
    }
}

// Helper function to launch the CUDA kernel for one iteration.
// The parameter 'which' (pointed to by &gpu_which) tells which GPU board is the current input.
void cuda_iteration(char* d_plate_0, char* d_plate_1, int n, int* which) {
    // Define block and grid dimensions (8x8 threads per block)
    dim3 blockDim(8, 8);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);
    
    printf("Launching kernel with grid: (%d, %d), block: (%d, %d)\n", 
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    
    // Launch the kernel alternating between the two GPU boards
    if (*which == 0) {
        update_plate_kernel<<<gridDim, blockDim>>>(d_plate_0, d_plate_1, n);
    } else {
        update_plate_kernel<<<gridDim, blockDim>>>(d_plate_1, d_plate_0, n);
    }
    
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    
    *which = !(*which); // Flip the board selector on the GPU
}

// Print the CPU plate (only if n is small enough to display)
void print_plate() {
    if (n < 60) {
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                printf("%d", (int) plate[which][i * (n + 2) + j]);
            }
            printf("\n");
        }
    } else {
        printf("Plate too large to print to screen\n");
    }
}

// Convert the CPU plate to a PNG image file.
void plate2png(char* filename) {
    unsigned char* img = (unsigned char*) malloc(n * n * sizeof(unsigned char));
    if (img == NULL) {
        fprintf(stderr, "Error allocating memory for image\n");
        exit(EXIT_FAILURE);
    }
    image_size_t sz;
    sz.width = n;
    sz.height = n; 
    
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            int pindex = i * (n + 2) + j;
            int index = (i - 1) * n + (j - 1);
            img[index] = (plate[which][pindex] > 0) ? 255 : 0;
        }
    }
    
    printf("Writing file %s\n", filename);
    write_png_file(filename, img, sz);
    printf("Done writing png\n");
    
    free(img);
    printf("Done freeing image memory\n");
}

int main() { 
    int M;
    char line[MAX_N + 2];  // Buffer for input lines (max n characters plus newline)
    
    // Read grid size (n) and number of iterations (M)
    if (scanf("%d %d", &n, &M) == 2) {
        // If n <= 0, use a default value for random generation.
        if (n <= 0) {
            n = 1000; // Use a reduced grid size for random generation
            printf("Using n = %d for random generation\n", n);
        }
        // Optional: enforce n <= MAX_N if needed.
        if(n > MAX_N) {
            printf("n (%d) exceeds MAX_N (%d), using MAX_N instead\n", n, MAX_N);
            n = MAX_N;
        }
        
        // Initialize CPU plates to 0 (note: the board is padded)
        memset(plate[0], 0, sizeof(char) * (n + 2) * (n + 2));
        memset(plate[1], 0, sizeof(char) * (n + 2) * (n + 2));
        
        // Fill the initial state.
        // If n==1000 (set via n<=0), use random generation.
        // Otherwise, assume the input file contains n lines of board data.
        if (n == 1000) {  
            for (int i = 1; i <= n; i++) {
                for (int j = 0; j < n; j++) {
                    plate[0][i * (n + 2) + j + 1] = (char)(rand() % 2);
                }
            }
        } else {
            for (int i = 1; i <= n; i++) {
                scanf("%s", line);
                for (int j = 0; j < n; j++) {
                    plate[0][i * (n + 2) + j + 1] = line[j] - '0';
                }
            }
        }
        
        // Allocate GPU memory for two boards of size (n+2) x (n+2)
        char *d_plate_0 = NULL;
        char *d_plate_1 = NULL;
        size_t plate_size = sizeof(char) * (n + 2) * (n + 2);
        printf("Allocating %zu bytes of GPU memory (total for two boards: %zu bytes)\n", 
               plate_size, 2 * plate_size);
        
        CUDA_CALL(cudaMalloc((void**)&d_plate_0, plate_size));
        CUDA_CALL(cudaMalloc((void**)&d_plate_1, plate_size));
        
        // Copy the starting CPU state (plate[0] and plate[1]) to GPU memory
        CUDA_CALL(cudaMemcpy(d_plate_0, plate[0], plate_size, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_plate_1, plate[1], plate_size, cudaMemcpyHostToDevice));
        
        int gpu_which = 0;  // This tracks which GPU board is current.
        
        printf("\nIteration %d:\n", 0);
        print_plate();
        
        // Main simulation loop on the CPU side invoking the GPU kernel each iteration
        for (int i = 0; i < M; i++) {
            cuda_iteration(d_plate_0, d_plate_1, n, &gpu_which);
            
            if (i < M - 1) {
                printf("\nIteration %d:\n", i + 1);
                // Copy the current GPU board back to the CPU for display (optional)
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
        
        // After simulation, copy the final state from the GPU back to the CPU.
        if (gpu_which == 0) {
            CUDA_CALL(cudaMemcpy(plate[0], d_plate_0, plate_size, cudaMemcpyDeviceToHost));
            which = 0;
        } else {
            CUDA_CALL(cudaMemcpy(plate[1], d_plate_1, plate_size, cudaMemcpyDeviceToHost));
            which = 1;
        }
        
        printf("\n\nFinal state:\n");
        char png_filename[] = "plate.png";
        plate2png(png_filename);
        print_plate();
        
        // Free GPU memory.
        CUDA_CALL(cudaFree(d_plate_0));
        CUDA_CALL(cudaFree(d_plate_1));
    }
    
    return 0;
}