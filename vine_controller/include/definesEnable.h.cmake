#ifndef ENABLE_DEFINES
#define ENABLE_DEFINES
#cmakedefine ELASTICITY 	                  //Disable/Enable breakdown timers inside controller
#cmakedefine BREAKDOWNS_CONTROLLER 	          //Disable/Enable breakdown timers inside controller
#cmakedefine BREAKDOWNS_VINETALK		      //Disable/Enable utils_breakdown_advance from vinetalk
#cmakedefine DATA_TRANSFER	                  //Disable/Enable prints for input-output data size
#cmakedefine REDUCTION_UNITS_FROM_EXEC_TIME	  //Reduction units according to execution time or statically defined as -1
#cmakedefine POWER_TIMER	 		 //Returns timers to be used for POWER-UTILZATION
#endif
