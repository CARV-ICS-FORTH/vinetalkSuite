#include "../include/darkGrayArgs.h"
#include "../include/cu_darkGray.h"
#include "VineLibUtilsGPU.h"

#include <chrono>

/* Kernel for the device */
__global__ void rgb_gray(const int width, const int height,
                         const unsigned char *inputImage,
                         unsigned char *darkGrayImage) {
  int x;
  int y;
//volatile int i; 
//for(i=0; i<100; ++i) {
  // calculate the thread index for both x, y, by the use of the dimension
  // of the block the id of the current block and the id of the thread
  y = blockDim.y * blockIdx.y + threadIdx.y;
  x = blockDim.x * blockIdx.x + threadIdx.x;

  // check if we are out of bounds
  if ((y * width + x) > (width * height)) {
    return;
  }
  // do the transformation
  float grayPix = 0.0f;
  float r = static_cast<float>(inputImage[(y * width) + x]);
  float g = static_cast<float>(inputImage[(width * height) + (y * width) + x]);
  float b =
      static_cast<float>(inputImage[(2 * width * height) + (y * width) + x]);
  grayPix = ((0.3f * r) + (0.59f * g) + (0.11f * b));
  grayPix = (grayPix * 0.6f) + 0.5f;
  darkGrayImage[(y * width) + x] = static_cast<unsigned char>(grayPix);
//}
}
/* Function that calls the kernel*/
void cu_darkGray(const int width, const int height,
                 const unsigned char *inputImageDev,
                 unsigned char *darkGrayImageDev) {
  int wBlock = static_cast<unsigned int>(ceil(width / static_cast<float>(32)));
  int hBlock = static_cast<unsigned int>(ceil(height / static_cast<float>(16)));
  dim3 dimGrid(wBlock, hBlock);
  dim3 dimBlock(32, 16);
  /* Kernel call */
  rgb_gray << <dimGrid, dimBlock>>>
      (width, height, inputImageDev, darkGrayImageDev);
}

/* Contains the code that is executed in Host*/
vine_task_state_e hostCode(vine_task_msg_s *vine_task) {
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::vector<void *> ioVector;
  darkGrayArgs *argsCuda;
  
  cout << "cu_darkgray execution in GPU." << endl;

  /* Get the actual arguments*/
  argsCuda = (darkGrayArgs *)vine_data_deref(vine_task->args.vine_data);

  /* Allocate memory in the device and transfer data */
  if (!Host2GPU(vine_task, ioVector)) {
    cerr << "Host2GPU" << endl;
    cerr << __FILE__ << " Failed at " << __LINE__ << endl;
    return (task_failed);
  }

#ifdef TIMERS_ENABLED
  start = std::chrono::system_clock::now();
#endif
  /* Call the function that calls the kernel */
  cu_darkGray(argsCuda->width, argsCuda->height, (unsigned char *)ioVector[0],
              (unsigned char *)ioVector[1]);
#ifdef TIMERS_ENABLED

  end = std::chrono::system_clock::now();

  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::chrono::duration<double, std::nano> elapsed_seconds = end - start;

  cout << "DarkGray kernel execution time: " << elapsed_seconds.count()
       << " nanosecs." << endl;
#endif
  /* Check for cuda errors*/
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    cerr << __FILE__ << " Failed at " << __LINE__ << endl;
    printf("Error: %s\n", cudaGetErrorString(err));
    return (task_failed);
  }

  /* Copy back the result  from GPU*/
  if (! (GPU2Host(vine_task, ioVector)) ) {
    cerr << "GPU2Host" << endl;
    cerr << __FILE__ << " Failed at " << __LINE__ << endl;
    return (task_failed);
  }

  /* Free device memory*/
  if (!GPUMemFree(ioVector)) {
    cerr << "GPUMemFree" << endl;
    cerr << __FILE__ << " Failed at " << __LINE__ << endl;
    return (task_failed);
  }
  /* Execution was successful*/
  return vine_task_stat(vine_task, 0);
}
/* register the function to the array for this .so*/
VINE_PROC_LIST_START()
VINE_PROCEDURE("darkGray", GPU, hostCode, sizeof(darkGrayArgs))
VINE_PROC_LIST_END()
