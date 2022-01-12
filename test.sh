#/bin/bash
export CUDA_VISIBLE_DEVICES=4
file_name='test.py'

if [ -f "${file_name}" ]; then
	chmod -x ./${file_name}
	python ${file_name} 
else
	echo " ${file_name} not find"
fi
