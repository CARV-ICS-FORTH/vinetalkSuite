#!/bin/bash

# ------ Compile IMG+NLP -------
## APPS
## go to img/nlp kai kanw make SOCKETS=y
## CONTROLLER
## go to process and make

# ------- Run --------
## Run this script 
## Be Carrefull numberOfGPUS = numOfAccelProcess
## So numOfProcs == should equal to GPUs from example config
## In JObGen run as an ordinary bench (in build make run


#run processes for multiple caffe instances

#numOfProcs = $1
#if [ $# -eq 0 ]
#then
#      echo "Add the number of processes."
#      exit
#fi

#cd processes/

#for (( i = 0 ; i < 4; i = i + 1 ))
#do
	#start multiple process (as many as the number of processors)
	#taskset -c $i+20 ./accelProcess 800$i $i &
#	taskset -c $((i+20)) ./accelProcess 800$i $i &
	#task$((i))=$!
	#task$i=$!
#done

taskset -c 3 ./processes/accelProcess 8000 0 &
taskset -c 4 ./processes/accelProcess 8001 1 &
taskset -c 11 ./processes/accelProcess 8002 3 &
taskset -c 12 ./processes/accelProcess 8003 4 &

#go to controller and strt him

./build/vine_controller example_config
controller=$!

kill -2 $controller

killall accelProcess

