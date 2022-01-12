#/bin/bash
export CUDA_VISIBLE_DEVICES=4
file_name='test.py'

if [ -f "${file_name}" ]; then
	chmod -x ./${file_name}
	python ${file_name} 
	# --test_model MobileNetV2_S \
	# --test_model_path ./checkpoint/mobilenetv2_s_RAdam_warmup_cosine_cutmix_labelsmooth_randaug_mixup/best_model.pt \
	# --test_dataset /media/data4/yg/DATASETS/zebracrossing_datasets_rgb/test_datasets_rgb.txt \ 
	# --batch_size 1 
	# --checkpoint_name mobilenetv2_s_RAdam_warmup_cosine_cutmix_labelsmooth_randaug_mixup_pretrained \
	# --optimizer RADAM --learning_rate 0.0001 --decay_type cosine_warmup --cutmix_alpha 1.0 --cutmix_prob 1.0 \
	# --label_smooth 0.1 --randaugment --mixup 0.2
else
	echo " ${file_name} not find"
fi