export CUDA_VISIBLE_DEVICES=0
echo $CUDA_VISIBLE_DEVICES

model=${PWD##*/}
echo $model

python3.6 ../main_split.py --log_info=config/log_info.yaml --path_info=config/path_info.cfg --network_info=config/network_info.cfg  --max_queue_size=40 --workers=5 --use_multiprocessing=True
