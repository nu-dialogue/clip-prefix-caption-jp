export CUDA_VISIBLE_DEVICES=6,7
python train.py --dataset_name coco \
                --rinna_gpt_name gpt_medium \
                --clip_model_name en_clip_b32 \
                --per_gpu_train_batch_size 24 \
                --per_gpu_eval_batch_size 24 \
                --lr 2e-5 \
                --save_every 1 \
                --mapping_type transformer \
                --n_gpu 2
                # --only_prefix \
