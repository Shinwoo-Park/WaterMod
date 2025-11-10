#! /bin/bash 

source ~/.bashrc 

conda activate markllm

date 

for dataset in "mbpp_plus" "gsm" "c4"
do 
    for model_path in "Qwen/Qwen2.5-1.5B"
    do 
        for watermark_algorithm in "SAM_MULTI" 
        do
            for temperature in 0.0 
            do
                python parse_generated_results.py --dataset $dataset --model_path $model_path --watermark_algorithm $watermark_algorithm --temperature $temperature
            done
        done
    done 
done 

for dataset in "mbpp_plus" "gsm" "c4"
do 
    for model_path in "Qwen/Qwen2.5-1.5B"
    do 
        for watermark_algorithm in "SAM_MULTI" 
        do
            for temperature in 0.0 
            do
                CUDA_VISIBLE_DEVICES=0 python evaluate_detection_performance.py --dataset $dataset --model_path $model_path --watermark_algorithm $watermark_algorithm --temperature $temperature
            done
        done
    done 
done 

cd evaluator/c4/
bash evaluate.sh
cd ../gsm/
bash evaluate.sh
cd ../evalplus/
bash evaluate.sh
cd ../../

date 

rm -rf __pycache__