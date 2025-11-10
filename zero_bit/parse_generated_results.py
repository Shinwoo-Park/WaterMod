import argparse
import pickle 
import json 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--watermark_algorithm', type=str, default='KGW')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=43)

    args = parser.parse_args()

    dataset = args.dataset
    llm = args.model_path.split("/")[-1]
    watermark_algorithm = args.watermark_algorithm
    temperature = str(args.temperature)
    seed = str(args.seed)

    # Load the results
    print("Loading the results...")
    file_name = f"experimental_results/generation_results_{dataset}_{llm}_{watermark_algorithm}_{seed}_{temperature}.pkl"
    with open(file_name, 'rb') as f:
        result = pickle.load(f)

    # Parse the results
    print("Parsing the results...")
    parsing_identifier = ""
    if dataset == "c4":
        parsing_identifier = "Continuation:"
    elif dataset == "gsm":
        parsing_identifier = "Answer:"
    elif dataset == "mbpp_plus":
        parsing_identifier = "Code:"
    else: 
        raise ValueError(f"Dataset {dataset} not supported for parsing.")

    parsed_watermarked_results = []
    for i in range(len(result['watermarked_contents'])):
        if parsing_identifier in result['watermarked_contents'][i]:
            parsed_watermarked_results.append(result['watermarked_contents'][i].split(parsing_identifier)[-1].strip())
        else:
            parsed_watermarked_results.append(result['watermarked_contents'][i])

    parsed_unwatermarked_results = []
    for i in range(len(result['unwatermarked_contents'])):
        if parsing_identifier in result['unwatermarked_contents'][i]:
            parsed_unwatermarked_results.append(result['unwatermarked_contents'][i].split(parsing_identifier)[-1].strip())
        else:
            parsed_unwatermarked_results.append(result['unwatermarked_contents'][i])

    # Save the parsed results
    print("Saving the parsed results...")
    if dataset == "mbpp_plus":
        watermarked_code_file_name = f"experimental_results/watermarked_solutions_{dataset}_{llm}_{watermark_algorithm}_{seed}_{temperature}.jsonl"
        unwatermarked_code_file_name = f"experimental_results/unwatermarked_solutions_{dataset}_{llm}_{watermark_algorithm}_{seed}_{temperature}.jsonl"
        watermarked_code = []
        unwatermarked_code = []
        with open(watermarked_code_file_name, 'r') as f:
            for line in f:
                watermarked_code.append(json.loads(line))
        with open(unwatermarked_code_file_name, 'r') as f:
            for line in f:
                unwatermarked_code.append(json.loads(line))
        parsed_watermarked_code = []
        parsed_unwatermarked_code = []
        for wc, pwr in zip(watermarked_code, parsed_watermarked_results):
            item = {} 
            item['task_id'] = wc['task_id']
            item['solution'] = pwr 
            parsed_watermarked_code.append(item)
        for uc, pur in zip(unwatermarked_code, parsed_unwatermarked_results):
            item = {}
            item['task_id'] = uc['task_id']
            item['solution'] = pur 
            parsed_unwatermarked_code.append(item)
        watermarked_code_file_name = f"evaluator/evalplus/parsed_watermarked_{dataset}_{llm}_{watermark_algorithm}_{seed}_{temperature}.jsonl"
        with open(watermarked_code_file_name, 'w') as f:
            for item in parsed_watermarked_code:
                f.write(json.dumps(item) + '\n')
        unwatermarked_code_file_name = f"evaluator/evalplus/parsed_unwatermarked_{dataset}_{llm}_{watermark_algorithm}_{seed}_{temperature}.jsonl"
        with open(unwatermarked_code_file_name, 'w') as f:
            for item in parsed_unwatermarked_code:
                f.write(json.dumps(item) + '\n')
    else: 
        parsed_watermarked_data = [] 
        parsed_unwatermarked_data = []
        for h, pw in zip (result['human_written_contents'], parsed_watermarked_results):
            item = {} 
            item['ground_truth'] = h
            item['generated'] = pw
            parsed_watermarked_data.append(item)
        for h, pu in zip (result['human_written_contents'], parsed_unwatermarked_results):
            item = {} 
            item['ground_truth'] = h
            item['generated'] = pu
            parsed_unwatermarked_data.append(item)
        watermarked_file_name = f"evaluator/{dataset}/parsed_watermarked_{dataset}_{llm}_{watermark_algorithm}_{seed}_{temperature}.jsonl"
        with open(watermarked_file_name, 'w') as f:
            for item in parsed_watermarked_data:
                f.write(json.dumps(item) + '\n')
        unwatermarked_file_name = f"evaluator/{dataset}/parsed_unwatermarked_{dataset}_{llm}_{watermark_algorithm}_{seed}_{temperature}.jsonl"
        with open(unwatermarked_file_name, 'w') as f:
            for item in parsed_unwatermarked_data:
                f.write(json.dumps(item) + '\n')