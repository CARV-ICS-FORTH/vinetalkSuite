# RGB to Gray
This is a simple example that does Grayscale Conversion and Darkening to an image. Our input images are RGB images, this means that every color is rendered adding together the three components representing Red, Green and Blue. The gray value of a pixel is given by weighting this three values and then summing them together. The formula is: gray = 0.3*R + 0.59*G + 0.11*B. To darken this grayscale image, the obtained value is multiplied by 0.6.

This application runs with two accelerator types CPU and GPU through the use of vine_talk.

# Folder layout
* include - Header files 
	- c_darkGray.h: The CPU version generated from c_darkGray.cpp to create .so.
	- cu_darkGray.h: The GPU version generated from cu_darkGray.cu to create .so.
	- darkGrayArgs.h: contains the Argument structure for the darkGray kernels.
	- CImg.h: Library used for importing/exporting images(http://www.cimg.eu/).

* input_images - Contains images for testing

* src -  Source code
	- c_darkGray.cpp: contains the CPU kernel, and the approproate function calls for preparing this execution. It is used to create the .so file.
  	- cu_darkGray.cu: contains the GPU kernel, and the approproate function calls for preparing this execution (cudaMalloc, cudaMemcpy, cudaFree). It is used to create the .so file.
	- vine_darkGray.cpp: contains the producer that calls darkGray to be executed in controller.
	- compare.cpp: Compares two images. Used for error checking.

* image_script.sh - For every input image, spawns a controller instance, launches a producer with an input image first with CPU and then with GPU acceleration.

# Building 
	- First build vine_talk  
 	- Type make

# Running  
	./bin/vine_darkGray <inputIimage.jpg> <outputImage.jpg> <accelerator type 1=GPU and 3=CPU>
  	( eg: ./bin/vine_darkGray input_images/image01.jpg out01.jpg 1 )
# Troubleshouting
	In case that you get an error saying: [CImg] *** CImgIOException *** [instance(0,0,0,0,(nill),non-shared)] etc...
	it means that imagemagic is not istalled. 
	For Centos do the following: 
		yum install gcc php-devel php-pear
		yum install ImageMagick ImageMagick-devel
	For Ubuntu do :
		sudo apt install php php-common gcc
		sudo apt install imagemagick
