class vineFunCallerInterface{
	public:
		float* vine_blackScholes();
		int vine_darkgrey();
		int vine_convolutionSeparable();
		int vine_histogram(int hist_version); //arg for histogram version 0=64, 1=256
		float* vine_montecarlo();

};
