import argparse
import json 
import re 

def extract_last_integer(text: str):
    cleaned_text = text.replace(',', '')
    numbers = re.findall(r"-?\b\d+\b", cleaned_text)
    if not numbers:
        return None
    return int(numbers[-1])

def is_correct(pred_text: str, gold: int):
    pred = extract_last_integer(pred_text)
    return pred == gold

def evaluate_all(predictions: list[str], gold_answers: list[str]):
    assert len(predictions) == len(gold_answers)

    total = len(predictions)
    correct = 0
    for pred_text, gold in zip(predictions, gold_answers):
        if is_correct(pred_text, extract_last_integer(gold)):
            correct += 1
    accuracy = correct / total
    return accuracy * 100

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='gsm')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--watermark_algorithm', type=str, default='SAM_MULTI')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=43)

    args = parser.parse_args()

    dataset = args.dataset
    llm = args.model_path.split('/')[-1]
    watermark_algorithm = args.watermark_algorithm
    temperature = str(args.temperature)
    seed = str(args.seed)

    # Load the parsed results
    parsed_watermarked = [] 
    parsed_unwatermarked = []
    with open(f'parsed_watermarked_{dataset}_{llm}_{watermark_algorithm}_{seed}_{temperature}.jsonl', 'r') as f:
        for line in f:
            parsed_watermarked.append(json.loads(line))
    with open(f'parsed_unwatermarked_{dataset}_{llm}_{watermark_algorithm}_{seed}_{temperature}.jsonl', 'r') as f:
        for line in f:
            parsed_unwatermarked.append(json.loads(line))

    ground_truth = []
    watermarked_generated = [] 
    unwatermarked_generated = [] 
    for inst in parsed_watermarked:
        ground_truth.append(inst['ground_truth'])
        watermarked_generated.append(inst['generated'])
    for inst in parsed_unwatermarked:
        unwatermarked_generated.append(inst['generated'])

    # Evaluate the performance of the watermarked and unwatermarked generations
    watermarked_performance = evaluate_all(watermarked_generated, ground_truth)
    unwatermarked_performance = evaluate_all(unwatermarked_generated, ground_truth)

    print(f"Watermarked Performance: {watermarked_performance:.2f}%")
    print(f"Unwatermarked Performance: {unwatermarked_performance:.2f}%")

    # Save the performance results 
    with open(f'parsed_watermarked_{dataset}_{llm}_{watermark_algorithm}_{seed}_{temperature}_eval_results.txt', 'w') as f:
        f.write(f"Watermarked Performance: {watermarked_performance:.2f}%\n")
    with open(f'parsed_unwatermarked_{dataset}_{llm}_{watermark_algorithm}_{seed}_{temperature}_eval_results.txt', 'w') as f:
        f.write(f"Unwatermarked Performance: {unwatermarked_performance:.2f}%\n")