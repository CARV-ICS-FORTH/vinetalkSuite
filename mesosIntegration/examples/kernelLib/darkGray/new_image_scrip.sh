#!/bin/bash
#delete output folder
rm -rf output_images
rm /dev/shm/test
#create folder for the outputs
mkdir output_images
#run controller
./../../vine_controller/bin/vine_controller lib/ &
cntr=$!
j=1

function foo()
{
	for (( i = 0 ; i < $1 ; i=i+1 ));do
		#run the application
		 bin/vine_darkGray input_images/hugeEmptyImage.png output_images/out$i.jpg $j &>/dev/null &
	done
	wait `jobs -p | grep -v $cntr`
}


for (( t = 1 ; t < 5 ; t=t+1 ));do
	time foo $t
done

kill $cntr
