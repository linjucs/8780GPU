#include "im2Gray.h"

#define BLOCK 32
#define TILE_WITH 32



/*
 
  Given an input image d_in, perform the grayscale operation 
  using the luminance formula i.e. 
  o[i] = 0.224f*r + 0.587f*g + 0.111*b; 
  
  Your kernel needs to check for boundary conditions 
  and write the output pixels in gray scale format. 

  you may vary the BLOCK parameter.
 
 */

__global__
void im2Gray_share(uchar4 *d_in, unsigned char *d_gray, int numRows, int numCols){
	 __shared__ uchar4 r[TILE_WITH][TILE_WITH];
  //__shared__ int c[TILE_WIDTH];
  int x = threadIdx.x+blockIdx.x*blockDim.x;
  int y = threadIdx.y+blockIdx.y*blockDim.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  if (y < numRows && x < numCols){
  int w = x+y*numRows;
  r[tx][ty] = d_in[w];
  uchar4 imagePoint = r[tx][ty];
  d_gray[w] = .299f*imagePoint.x + .587f*imagePoint.y  + .114f*imagePoint.z;
  __syncthreads();
  }

}


__global__ 
void im2Gray(uchar4 *d_in, unsigned char *d_gray, int numRows, int numCols){

 /*
   Your kernel here: Make sure to check for boundary conditions
  */
		
  int x = threadIdx.x + blockIdx.x*blockDim.x;
  int y = threadIdx.y + blockIdx.y*blockDim.y;
  if (y < numRows && x < numCols){
	int pointIndex = y*numRows + x;
	uchar4 imagePoint = d_in[pointIndex];
	d_gray[pointIndex] = .299f*imagePoint.x + .587f*imagePoint.y  + .114f*imagePoint.z;
  }
}




void launch_im2gray(uchar4 *d_in, unsigned char* d_grey, size_t numRows, size_t numCols){
    // configure launch params here 
    int x_thread = TILE_WITH;
    int y_thread = TILE_WITH;
    int grid_x = numCols/x_thread;
    int grid_y = numRows/y_thread; 
    dim3 block(x_thread,y_thread,1);
    dim3 grid(ceil(grid_x),ceil(grid_y), 1);

    im2Gray<<<grid,block>>>(d_in, d_grey, numRows, numCols);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    
}





