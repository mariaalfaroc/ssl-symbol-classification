#!/bin/bash


echo "Converting PNG files to JPG"
for u in $(ls MTH1000/img/*.png);
do
	convert $u "${u%.*}".png 
	rm $u
done
echo "Done!"
