CUDA_VISIBLE_DEVICES=0 python get_feature.py --model_name resnet18 --cache_dir /cluster/home/hjjiang/PR-project/data/new_split_data/ &
CUDA_VISIBLE_DEVICES=1 python get_feature.py --model_name resnet50 --cache_dir /cluster/home/hjjiang/PR-project/data/new_split_data/ &
CUDA_VISIBLE_DEVICES=2 python get_feature.py --model_name resnet101 --cache_dir /cluster/home/hjjiang/PR-project/data/new_split_data/ &
CUDA_VISIBLE_DEVICES=3 python get_feature.py --model_name resnet152 --cache_dir /cluster/home/hjjiang/PR-project/data/new_split_data/