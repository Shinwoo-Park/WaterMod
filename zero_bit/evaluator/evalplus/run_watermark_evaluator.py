import argparse
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str)
    
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--watermark_algorithm', type=str, default='KGW')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=43)

    args = parser.parse_args()

    dataset = args.dataset
    llm = args.model_path.split('/')[-1]
    watermark_algorithm = args.watermark_algorithm
    temperature = str(args.temperature)
    seed = str(args.seed)

    watermarked_sanitized_args = ['--samples', f'parsed_watermarked_{dataset}_{llm}_{watermark_algorithm}_{seed}_{temperature}.jsonl']
    unwatermarked_sanitized_args = ['--samples', f'parsed_unwatermarked_{dataset}_{llm}_{watermark_algorithm}_{seed}_{temperature}.jsonl']
    watermarked_evaluate_args = ['--dataset', dataset, '--samples', f'parsed_watermarked_{dataset}_{llm}_{watermark_algorithm}_{seed}_{temperature}-sanitized.jsonl']
    unwatermarked_evaluate_args = ['--dataset', dataset, '--samples', f'parsed_unwatermarked_{dataset}_{llm}_{watermark_algorithm}_{seed}_{temperature}-sanitized.jsonl']

    watermarked_sanitized_cmd = ['python', 'sanitize.py'] + watermarked_sanitized_args
    unwatermarked_sanitized_cmd = ['python', 'sanitize.py'] + unwatermarked_sanitized_args

    watermarked_evaluate_cmd = ['python', 'evaluate.py'] + watermarked_evaluate_args
    unwatermarked_evaluate_cmd = ['python', 'evaluate.py'] + unwatermarked_evaluate_args

    subprocess.run(watermarked_sanitized_cmd)
    subprocess.run(unwatermarked_sanitized_cmd)
    subprocess.run(watermarked_evaluate_cmd)
    subprocess.run(unwatermarked_evaluate_cmd)
