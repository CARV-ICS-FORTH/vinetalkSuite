/* Kernel for the device */
__kernel void rgb_gray(const int width, const int height,
		const __global unsigned char *inputImage,
		__global unsigned char *darkGrayImage) {
	int x;
	int y;
	//volatile int i;
	//for(i=0; i<100; ++i) {
	// calculate the thread index for both x, y, by the use of the dimension
	// of the block the id of the current block and the id of the thread

	x = get_global_id(0);
	y = get_global_id(1);

	// check if we are out of bounds
	if ((y * width + x) >= (width * height)) {
		return;
	}
	// do the transformation
	float grayPix = 0.0f;
	float r = (float) (inputImage[(y * width) + x]);
	float g = (float) (inputImage[(width * height) + (y * width) + x]);
	float b =
		(float) (inputImage[(2 * width * height) + (y * width) + x]);
	grayPix = ((0.3f * r) + (0.59f * g) + (0.11f * b));
	grayPix = (grayPix * 0.6f) + 0.5f;
	darkGrayImage[(y * width) + x] = (unsigned char) (grayPix);
	//}
}