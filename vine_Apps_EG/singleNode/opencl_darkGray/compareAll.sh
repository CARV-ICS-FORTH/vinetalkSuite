
output=$1
if [ $# -eq 0 ]
then
	echo "Using directory 'output' for processed images"
	output='output'
else
	echo "Using directory '${output}' for processed images"
fi

echo
echo

./bin/compare input_images/image00.png ${output}/image00.jpg
./bin/compare input_images/image01.jpg ${output}/image01.jpg
./bin/compare input_images/image02.jpg ${output}/image02.jpg
./bin/compare input_images/image03.jpg ${output}/image03.jpg
./bin/compare input_images/image04.jpg ${output}/image04.jpg
./bin/compare input_images/image05.jpg ${output}/image05.jpg
./bin/compare input_images/image06.jpg ${output}/image06.jpg
./bin/compare input_images/image07.jpg ${output}/image07.jpg
./bin/compare input_images/image08.jpg ${output}/image08.jpg
./bin/compare input_images/image09.png ${output}/image09.jpg
