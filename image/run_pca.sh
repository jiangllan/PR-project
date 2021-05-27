CUDA_VISIBLE_DEVICES=0 python pca.py --model_name resnet18 --log_name log_resnet18_n_components_from_512_to_16.txt &
#CUDA_VISIBLE_DEVICES=1 python pca.py --model_name resnet101 --log_name log_resnet101_n_components_from_2048_to_16.txt &
#CUDA_VISIBLE_DEVICES=2 python pca.py --model_name resnet152 --log_name log_resnet152_n_components_from_2048_to_16.txt