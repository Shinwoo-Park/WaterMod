import os
import argparse
import pickle 
import json
from evalplus.data import get_mbpp_plus
import numpy as np
import random
import time 
import gc 
import torch
import tqdm
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

def clear_cuda_memory():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect() 
    torch.cuda.reset_accumulated_memory_stats() 

# Set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_c4_prompt(context):
    return f"""Continue the given text naturally and coherently. Please write only the continuation under "Continuation:", with no explanation.
    Given Text: {context}
    Continuation:
    """

def generate_gsm_prompt(question):
	return f"""Solve the following math problem step by step. First, provide a detailed step-by-step solution under "Solution:". Then, provide only the final answer under "Answer:", with no explanation.
    Question: {question}
    Solution:
    Answer:
    """

def generate_mbpp_prompt(problem_description):
    return f"""Here is a Python programming problem. Implement a Python function based on the given problem description. Please write only the code for the problem description under "Code:", with no explanation.
    Problem Description: {problem_description}
    Code:
    """

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--watermark_algorithm', type=str)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=43)

    args = parser.parse_args()

    ### Watermark Setup 

    # Device 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Set seed
    print(f"Seed: {args.seed}")
    set_seed(args.seed)

    model_name = args.model_path.split('/')[-1]
    print(f"Model: {model_name}")
    print(f"Watermark Algorithm: {args.watermark_algorithm}")
    llm_model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
    llm_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm_vocab_size = llm_tokenizer.vocab_size    
    max_num_tokens = 0
    if args.dataset == 'c4':
        max_num_tokens = 400
    elif args.dataset == 'gsm':
        max_num_tokens = 600
    elif args.dataset == 'mbpp_plus':
        max_num_tokens = 600
    else: 
        raise ValueError(f"Invalid dataset: {args.dataset}")

    if args.temperature == 0.0:
        transformers_config = TransformersConfig(
                                    model=llm_model,
                                    tokenizer=llm_tokenizer,
                                    vocab_size=llm_vocab_size,
                                    device=device,
                                    max_new_tokens=max_num_tokens,
                                    do_sample=False)
    else: 
        transformers_config = TransformersConfig(
                                    model=llm_model,
                                    tokenizer=llm_tokenizer,
                                    vocab_size=llm_vocab_size,
                                    device=device,
                                    max_new_tokens=max_num_tokens,
                                    temperature=args.temperature,
                                    do_sample=True)
    
    myWatermark = AutoWatermark.load(args.watermark_algorithm, 
                                    algorithm_config=f'config/{args.watermark_algorithm}.json',
                                    transformers_config=transformers_config)

    # Load dataset
    print(f"Dataset: {args.dataset}")
    human_written = []
    prompts = []
    sub_prompts = []
    task_ids = [] 
    if args.dataset == 'c4': 
        with open('watermark_datasets/c4_500.jsonl', 'r') as f:
            lines = f.readlines()
            lines = [json.loads(line) for line in lines]
            for line in lines:
                human_written.append(line['natural_text'])
                prompts.append(line['prompt'])
    elif args.dataset == 'gsm':
        with open('watermark_datasets/gsm.jsonl', 'r') as f:
            lines = f.readlines()
            lines = [json.loads(line) for line in lines]
            for line in lines:
                human_written.append(line['answer'].split("####")[-1].strip())
                prompts.append(line['question'])
    elif args.dataset == 'mbpp_plus':
        for task_id, problem in get_mbpp_plus().items():
            human_written.append(problem['canonical_solution'])
            prompts.append(problem['prompt'])
            task_ids.append(task_id)          
    else: 
        raise ValueError(f"Invalid dataset: {args.dataset}")

    # Prepare prompt for watermarking
    watermarking_prompts = [] 
    if args.dataset == 'c4':
        for context in prompts:
            watermarking_prompts.append(generate_c4_prompt(context))
    elif args.dataset == 'gsm':
        for question in prompts:
            watermarking_prompts.append(generate_gsm_prompt(question))
    elif args.dataset == 'mbpp_plus':
        for problem_description in prompts:
            watermarking_prompts.append(generate_mbpp_prompt(problem_description))
    print(f"Number of prompts: {len(watermarking_prompts)}")
    print("Watermarking Prompts: " + watermarking_prompts[0])

    ### Watermarking Experiment 
    if not os.path.exists('experimental_results'):
        os.makedirs('experimental_results')

    # Human-written z-score
    human_written_detect_results = []
    for hr in human_written:
        try:
            detect_result = myWatermark.detect_watermark(hr)
            clear_cuda_memory()
            human_written_detect_results.append(detect_result['score'])
        except Exception as e:
            human_written_detect_results.append("ERROR: " + str(e))

    # Watermarked LLM-generated content
    watermarked_content = []
    # Watermarked LLM-generated content z-score
    watermarked_detect_results = []
    # Unwatermarked LLM-generated content
    unwatermarked_content = []
    # Unwatermarked LLM-generated content z-score
    unwatermarked_detect_results = []

    # For Time Analysis
    watermark_insertion_time = []
    watermark_detection_time = []
    non_watermarked_generation_time = []
    non_watermarked_detection_time = []

    # Watermark Insertion & Extraction
    print("Watermarking Experiment")
    for watermark_prompt in tqdm.tqdm(watermarking_prompts):
        try: 
            start_time = time.time()
            watermarked_text = myWatermark.generate_watermarked_text(watermark_prompt)
            watermark_insertion_time.append(time.time() - start_time)
            watermarked_content.append(watermarked_text)
            clear_cuda_memory()

            start_time = time.time()
            detect_result = myWatermark.detect_watermark(watermarked_text)
            watermark_detection_time.append(time.time() - start_time)
            watermarked_detect_results.append(detect_result['score'])
            clear_cuda_memory()

            start_time = time.time()
            unwatermarked_text = myWatermark.generate_unwatermarked_text(watermark_prompt)
            non_watermarked_generation_time.append(time.time() - start_time)
            unwatermarked_content.append(unwatermarked_text)
            clear_cuda_memory()

            start_time = time.time()
            detect_result = myWatermark.detect_watermark(unwatermarked_text)
            non_watermarked_detection_time.append(time.time() - start_time)
            unwatermarked_detect_results.append(detect_result['score'])
            clear_cuda_memory()

        except Exception as e:
            watermark_insertion_time.append("ERROR: " + str(e))
            watermark_detection_time.append("ERROR: " + str(e))
            watermarked_content.append("ERROR: " + str(e))
            watermarked_detect_results.append("ERROR: " + str(e))
            non_watermarked_generation_time.append("ERROR: " + str(e))
            non_watermarked_detection_time.append("ERROR: " + str(e))
            unwatermarked_content.append("ERROR: " + str(e))
            unwatermarked_detect_results.append("ERROR: " + str(e))

    print("Save Results")
    generation_results = {}
    # Prompt 
    generation_results['prompts'] = watermarking_prompts
    # Z-score Analysis
    generation_results['human_written_z-score'] = human_written_detect_results
    generation_results['watermarked_z-score'] = watermarked_detect_results
    generation_results['unwatermarked_z-score'] = unwatermarked_detect_results
    # Time Analysis
    generation_results['watermarked_generation_time'] = watermark_insertion_time
    generation_results['watermarked_detection_time'] = watermark_detection_time
    generation_results['unwatermarked_generation_time'] = non_watermarked_generation_time
    generation_results['unwatermarked_detection_time'] = non_watermarked_detection_time
    # Genereated Content
    generation_results['human_written_contents'] = human_written
    generation_results['watermarked_contents'] = watermarked_content
    generation_results['unwatermarked_contents'] = unwatermarked_content
    generation_results_file = f"experimental_results/generation_results_{args.dataset}_{model_name}_{args.watermark_algorithm}_{args.seed}_{args.temperature}.pkl"
    with open(generation_results_file, 'wb') as f:
        pickle.dump(generation_results, f)

    # For MBPP+
    watermarked_solutions = []
    for task_id, code in zip(task_ids, watermarked_content):
        watermarked_solutions.append({'task_id': task_id, 'solution': code})
    unwatermarked_solutions = []
    for task_id, code in zip(task_ids, unwatermarked_content):
        unwatermarked_solutions.append({'task_id': task_id, 'solution': code})
    if args.dataset == 'mbpp_plus':
        with open(f"experimental_results/watermarked_solutions_{args.dataset}_{model_name}_{args.watermark_algorithm}_{args.seed}_{args.temperature}.jsonl", 'w') as f:
            for item in watermarked_solutions:
                f.write(json.dumps(item) + '\n')
        with open(f"experimental_results/unwatermarked_solutions_{args.dataset}_{model_name}_{args.watermark_algorithm}_{args.seed}_{args.temperature}.jsonl", 'w') as f:
            for item in unwatermarked_solutions:
                f.write(json.dumps(item) + '\n')