#! /bin/bash 

source ~/.bashrc 

conda activate markllm

date 

for dataset in "mbpp_plus"
do 
    for model_path in "Qwen/Qwen2.5-1.5B"
    do 
        for watermark_algorithm in "SAM" 
        do
            for temperature in 0.0 
            do
                python run_watermark_evaluator.py --dataset $dataset --model_path $model_path --watermark_algorithm $watermark_algorithm --temperature $temperature
            done
        done
    done 
done 

date 

rm -rf __pycache__