import argparse
import json 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

def compute_perplexity(model, tokenizer, text, device="cuda"):
    inputs    = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"].long().to(device)
    
    if input_ids.size(1) < 2:
        return None

    labels = input_ids.clone()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            return_dict=True
        )
        loss = outputs.loss.item()

    return math.exp(loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='c4')
    
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--watermark_algorithm', type=str, default='SAM_MULTI')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=43)

    args = parser.parse_args()

    dataset = args.dataset
    llm = args.model_path.split("/")[-1]
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

    # Load the model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)

    # Calculate perplexity for the watermarked and unwatermarked generations
    watermarked_perplexities = []
    unwatermarked_perplexities = []

    for generated_text in watermarked_generated:
        ppl = compute_perplexity(model, tokenizer, generated_text, device)
        watermarked_perplexities.append(ppl)
    for generated_text in unwatermarked_generated:
        ppl = compute_perplexity(model, tokenizer, generated_text, device)
        unwatermarked_perplexities.append(ppl)

    # Mean perplexity
    watermarked_perplexities = [ppl for ppl in watermarked_perplexities if ppl is not None]
    watermarked_performance = sum(watermarked_perplexities) / len(watermarked_perplexities)
    unwatermarked_perplexities = [ppl for ppl in unwatermarked_perplexities if ppl is not None]
    unwatermarked_performance = sum(unwatermarked_perplexities) / len(unwatermarked_perplexities)

    print(f"Watermarked Performance: {watermarked_performance:.2f}")
    print(f"Unwatermarked Performance: {unwatermarked_performance:.2f}")

    # Save the performance results 
    with open(f'parsed_watermarked_{dataset}_{llm}_{watermark_algorithm}_{seed}_{temperature}_eval_results.txt', 'w') as f:
        f.write(f"Watermarked Performance: {watermarked_performance:.2f}\n")
    with open(f'parsed_unwatermarked_{dataset}_{llm}_{watermark_algorithm}_{seed}_{temperature}_eval_results.txt', 'w') as f:
        f.write(f"Unwatermarked Performance: {unwatermarked_performance:.2f}\n")