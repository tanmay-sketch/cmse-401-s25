#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include "png_util.h"
#include <cuda.h>
#define MAX_N 20000
#define CUDA_CALL(x) {cudaError_t cuda_error__ = (x); if (cuda_error__) std::cout << "CUDA error: " << #x << " returned " << cudaGetErrorString(cuda_error__) << std::endl;}

char plate[2][(MAX_N + 2) * (MAX_N + 2)];
int which = 0;
int n;

__global__ void iteration_kernel(char* d_plate, int n, int which) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1; // Fixed to use threadIdx.x for columns

    if(i <= n && j <= n) {
        int stride = n + 2;
        int index = i * stride + j;
        int plate_size = (n + 2) * (n + 2);
        char* curr = d_plate + which * plate_size;
        char* next = d_plate + ((which ^ 1) * plate_size);
        int num = curr[index - stride - 1] + curr[index - stride] + curr[index - stride + 1] +
                  curr[index - 1] + curr[index + 1] +
                  curr[index + stride - 1] + curr[index + stride] + curr[index + stride + 1];
        if(curr[index])
            next[index] = (num == 2 || num == 3) ? 1 : 0;
        else
            next[index] = (num == 3); 
    }
}

void print_plate(){
    if (n < 60) {
        for(int i = 1; i <= n; i++){
            for(int j = 1; j <= n; j++){
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

    for(int i = 1; i <= n; i++){
        for(int j = 1; j <= n; j++){
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
        if (n > 0) {
                memset(plate[0], 0, sizeof(char) * (n + 2) * (n + 2));
                memset(plate[1], 0, sizeof(char) * (n + 2) * (n + 2));
                for(int i = 1; i <= n; i++){
                    scanf("%s", line);
                    for(int j = 0; j < n; j++){
                        plate[0][i * (n + 2) + j + 1] = line[j] - '0';
                    }
                }
        } else {
        n = MAX_N; 
        for(int i = 1; i <= n; i++) 
                for(int j = 0; j < n; j++) 
                    plate[0][i * (n+2) +j + 1] = (char) rand() % 2;
        }

        // ------- GPU initialization ------- 
        int plate_size = (n + 2) * (n + 2) * sizeof(char);
        int total_size = 2 * plate_size;
        char* d_plate;

        // -------- Allocating and copying to GPU ---------
        CUDA_CALL(cudaMalloc((void**)&d_plate, total_size));
        CUDA_CALL(cudaMemcpy(d_plate, plate[0], plate_size, cudaMemcpyHostToDevice));
        dim3 block(16,16); // defines a block of 16x16 threads
        dim3 grid((n + block.x - 1)/block.x, (n + block.y - 1)/block.y); // calculates how many blocks are needed in each dimension to cover all cells in nxn grid
        
        for(int i = 0; i < M; i++) {
            if(n < 60) {
                CUDA_CALL(cudaMemcpy(plate[which], d_plate + which * plate_size, plate_size, cudaMemcpyDeviceToHost));
                printf("\nIteration %d:\n",i);
                print_plate();
            }
            iteration_kernel<<<grid, block>>>(d_plate, n, which);
            CUDA_CALL(cudaGetLastError()); // checks for any error
            CUDA_CALL(cudaDeviceSynchronize()); //ensures kernel has completed its work
            which ^= 1; //switches the board using xor so that the next iteration will update the correct board
        }
        CUDA_CALL(cudaMemcpy(plate[which], d_plate + which * plate_size, plate_size, cudaMemcpyDeviceToHost));
        
        char filename[] = "plate.png";  // Create a modifiable string
        plate2png(filename);
        
        print_plate();
        CUDA_CALL(cudaFree(d_plate));
    }
    return 0;
}
