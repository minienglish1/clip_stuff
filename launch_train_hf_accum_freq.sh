export CUDA_VISIBLE_DEVICES=1
source venv/bin/activate
python train-OpenAI-CLIP-ViT-L-14_hf_3_accum_freq.py \
	--image_text_dirs "/mnt/storage/cache_ComicArtCommunity" \
	--accum_freq 2 \
	--deterministic
