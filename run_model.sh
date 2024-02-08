for dataset_name in "ml-1m" "OR" "Office" "Food" "Movies"
do
    CUDA_VISIBLE_DEVICES=0 nohup python main.py -d $dataset_name --gpu_id=0 >> log_results/tedrec_$dataset_name\.log &
done