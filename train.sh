#/bin/bash
export CUDA_VISIBLE_DEVICES=4
file_name='main.py'

if [ -x "${file_name}" ]; then
	python ${file_name} --checkpoint_name mobilenetv2_s_RAdam_warmup_cosine_cutmix_labelsmooth_randaug_mixup_2.3M_pretrained \
	--optimizer RADAM --learning_rate 0.0001 --decay_type cosine_warmup --cutmix_alpha 1.0 --cutmix_prob 1.0 \
	--label_smooth 0.1 --randaugment --mixup 0.2
else
	echo " ${file_name} not find"
fi