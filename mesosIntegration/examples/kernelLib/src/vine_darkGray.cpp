#include <iostream>
#include <iomanip>
#include <stdlib.h>

/* Include for Image processing */
#include "CImg.h"
#include <chrono>
/* Includes from stelios*/

#include "vine_talk.h"
#include "darkGrayArgs.h"
#include "statisticsDefineEnable.h"
#include "vineFunCaller.hpp"
using std::cout;
using std::endl;
using std::cerr;
using cimg_library::CImg;

//int main(int argc, char *argv[])
int vineFunCallerInterface::vine_darkgrey()
{
  /* Size of image*/
  size_t sizeImage, outImageSize;
  /* Accelerator type */
  vine_accel **allAccels;
  /* Counter for the accelerators of a specific type*/
  int accelsCount;
  /* The accelerator to use */
  vine_accel *accelInUse;

  /* Get the arguments which are:
   * an input image - image for processing.
   * an output image - place the result from RGB2Gray.
   * accelerator type - the accelerator that RGB2Gray
                        is going to executed.
  */ /*
  if (argc != 4) {
    cerr << "Usage: " << argv[0] << " <input filename>"
         << " <output filename> "
         << "GPU = 1"
         << " CPU = 3";
    return 1;
  }*/
  /* Initialize vine_talk*/
  vine_talk_init();
#ifdef STATISTICS_TIMERS_ENABLED
  std::chrono::time_point<std::chrono::system_clock> startCreatingImg,
      endCreatingImg;
  startCreatingImg = std::chrono::system_clock::now();
#endif

  /* Load the input image */
  //CImg<unsigned char> inputImage = CImg<unsigned char>(argv[1]);
  CImg<unsigned char> inputImage = CImg<unsigned char>("/home/gelagotis/vac4mesos/input_images/image06.jpg");

#ifdef STATISTICS_TIMERS_ENABLED
  endCreatingImg = std::chrono::system_clock::now();
  std::chrono::duration<double, std::nano> elapsedInit = endCreatingImg - startCreatingImg;
  cout << endl;
  cout << "------------------------Breakdown Stats--------------------------"
       << endl;
  cout << "Creating Image with size "
       << inputImage.size() * sizeof(*inputImage.data())
       << " took : " << elapsedInit.count() << " nanosec." << endl;
#endif

  /* Read the accelerator type provided from user */
  //vine_accel_type_e accelType = (vine_accel_type_e)atoi(argv[3]);
  vine_accel_type_e accelType = (vine_accel_type_e)1;
  if (inputImage.spectrum() != 3) {
    cerr << "The input must be a color image." << endl;
    return 1;
  }

  /* Create a task id by the use of accelerator and name of function */
  vine_proc *process_id = vine_proc_get(accelType, "darkGray");

  /* Get the actual data (type cast to vine_argument) */
  darkGrayArgs dGrayArgs;

  /* Pass the image width/height to the struct */
  dGrayArgs.width = inputImage.width();    // image width
  dGrayArgs.height = inputImage.height();  // image height
  vine_buffer_s args[1] = {VINE_BUFFER(&dGrayArgs, sizeof(darkGrayArgs))};

  /* Find the size of the image */
  sizeImage = inputImage.size() * sizeof(*inputImage.data());

  vine_buffer_s inputs[1] = {VINE_BUFFER(inputImage.data(), sizeImage)};
  
  /*use the default values for dimension=1, spectrum=1*/
  CImg<unsigned char> outImage =
      CImg<unsigned char>(inputImage.width(), inputImage.height());
  /* Find the size of the output image != from input image size
   * due to spectrum. At input image spectrum is 3 because is colored 
   * output image's spectrum is 1 because is grayscaled
   * */  
  outImageSize = outImage.size() * sizeof(*outImage.data());

  vine_buffer_s outputs[1] = {VINE_BUFFER(outImage.data(), outImageSize)};

  /*get a dedicated channel*/
  accelInUse = vine_accel_acquire_type(accelType);

  /* Issue task to accelerator. */
  vine_task *task =
      vine_task_issue(accelInUse, process_id, args, 1, inputs, 1, outputs);

  /* Wait for task or exit if it fails */
  if (vine_task_wait(task) == task_completed) {
    cout << "Image processing has been completed succesfully!" << endl;
  } else {
    cout << "Image processing has FAILED!" << endl;
    return -1;
  }

  /* Save output */
  //outImage.save(argv[2]);
  outImage.save("/home/gelagotis/vac4mesos/output_images/greyImage.jpg");
  
  vine_task_free(task);
  /* Release data buffers */
  vine_accel_release(&accelInUse);
  /* close vine_talk*/
  vine_talk_exit();
  cout<<"Exiting vine DarkGrey"<< endl;
  return 0;
}
