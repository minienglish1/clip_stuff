#export CUDA_VISIBLE_DEVICES=0
#	--num_cpu_threads_per_process 2 \
#	--config_file "FSDP_0_fp16_NO_SHARD.yaml" \
source venv/bin/activate
accelerate launch \
	--config_file "default_config.yaml" \
	clip_finetune_3.py \
	--image_text_dirs "/mnt/storage/cache_ComicArtCommunity/" \
	--epochs 50 \
	--lr 5e-7 \
	--batch_size 220
