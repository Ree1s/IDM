
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12392 --use_env idm_main.py -p train -c config/ffhq_liifsr3_scaler_16_128.json -r checkpoints/face 

# val
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12392 --use_env idm_main.py -p val -c config/ffhq_liifsr3_scaler_16_128.json -r checkpoints/face
