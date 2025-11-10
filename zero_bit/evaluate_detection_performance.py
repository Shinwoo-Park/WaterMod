import argparse
import pickle
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import sklearn.metrics as metrics
import numpy as np

def compute_entropies(model, tokenizer, text, tau=1.0, device="cuda"):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(device)
        if input_ids.size(1) < 2:
            return [], []
        
        with torch.no_grad():
            outputs = model(input_ids, return_dict=True)
            logits = outputs.logits  # (1, seq_len, vocab_size)

        shannon_entropies = []
        spike_entropies   = []
        for i in range(1, input_ids.size(1)):
            dist_logits = logits[0, i-1]
            probs       = F.softmax(dist_logits, dim=-1)

            # Shannon entropy 
            H_shannon = -(probs * torch.log(probs + 1e-12)).sum().item()
            # Spike entropy  
            H_spike   = (probs / (1.0 + tau * probs)).sum().item()

            shannon_entropies.append(H_shannon)
            spike_entropies.append(H_spike)

        return shannon_entropies, spike_entropies

    except Exception as e:
        print(f"[Warning] {e}")
        return [], []

def get_roc_aur(human_z, machine_z):
    assert len(human_z) == len(machine_z)

    baseline_z_scores = np.array(human_z)
    watermark_z_scores = np.array(machine_z)
    all_scores = np.concatenate([baseline_z_scores, watermark_z_scores])

    baseline_labels = np.zeros_like(baseline_z_scores)
    watermarked_labels = np.ones_like(watermark_z_scores)
    all_labels = np.concatenate([baseline_labels, watermarked_labels])

    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_scores, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    
    return roc_auc, fpr, tpr, thresholds

def replace_non_floats(lst):
    return [x if isinstance(x, float) else 0 for x in lst]

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

    # Load the model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)

    # Load the results
    print("Loading the results...")
    file_name = f"experimental_results/generation_results_{dataset}_{llm}_{watermark_algorithm}_{seed}_{temperature}.pkl"
    with open(file_name, 'rb') as f:
        result = pickle.load(f)

    # Evaluate the results 
    result["watermarked_z-score"] = replace_non_floats(result["watermarked_z-score"])
    result["unwatermarked_z-score"] = replace_non_floats(result["unwatermarked_z-score"])
    result["human_written_z-score"] = replace_non_floats(result["human_written_z-score"])
    mean_watermarked_z_score = np.mean(result["watermarked_z-score"])
    mean_unwatermarked_z_score = np.mean(result["unwatermarked_z-score"])
    human_z_score = replace_non_floats(result["human_written_z-score"])
    mean_human_z_score = np.mean(human_z_score)
    roc_aur, fpr, tpr, thresholds = get_roc_aur(human_z_score, result["watermarked_z-score"]) 

    mean_watermarked_geneation_time = np.mean(result["watermarked_generation_time"])
    mean_unwatermarked_geneation_time = np.mean(result["unwatermarked_generation_time"])
    mean_watermarked_detection_time = np.mean(result["watermarked_detection_time"])
    mean_unwatermarked_detection_time = np.mean(result["unwatermarked_detection_time"])

    watermarked_contents = result["watermarked_contents"]
    unwatermarked_contents = result["unwatermarked_contents"]

    watermarked_shannon_entropies = []
    unwatermarked_shannon_entropies = []
    watermarked_spike_entropies = []
    unwatermarked_spike_entropies = []

    for content in watermarked_contents:
        shannon_entropy, spike_entropy = compute_entropies(model, tokenizer, content, tau=1.0, device=device) 
        watermarked_shannon_entropies.append(np.mean(shannon_entropy))
        watermarked_spike_entropies.append(np.mean(spike_entropy))

    for content in unwatermarked_contents:
        shannon_entropy, spike_entropy = compute_entropies(model, tokenizer, content, tau=1.0, device=device) 
        unwatermarked_shannon_entropies.append(np.mean(shannon_entropy))
        unwatermarked_spike_entropies.append(np.mean(spike_entropy))

    mean_watermarked_shannon_entropy = np.mean(watermarked_shannon_entropies)
    mean_unwatermarked_shannon_entropy = np.mean(unwatermarked_shannon_entropies)
    mean_watermarked_spike_entropy = np.mean(watermarked_spike_entropies)
    mean_unwatermarked_spike_entropy = np.mean(unwatermarked_spike_entropies)

    # Save the results
    evaluation_results = {}
    evaluation_results["mean_watermarked_z-score"] = mean_watermarked_z_score
    evaluation_results["mean_unwatermarked_z-score"] = mean_unwatermarked_z_score
    evaluation_results["mean_human_z-score"] = mean_human_z_score
    evaluation_results["AUROC"] = roc_aur * 100 
    evaluation_results["mean_watermarked_generation_time"] = mean_watermarked_geneation_time
    evaluation_results["mean_unwatermarked_generation_time"] = mean_unwatermarked_geneation_time
    evaluation_results["mean_watermarked_detection_time"] = mean_watermarked_detection_time
    evaluation_results["mean_unwatermarked_detection_time"] = mean_unwatermarked_detection_time
    evaluation_results["mean_watermarked_shannon_entropy"] = mean_watermarked_shannon_entropy
    evaluation_results["mean_unwatermarked_shannon_entropy"] = mean_unwatermarked_shannon_entropy
    evaluation_results["mean_watermarked_spike_entropy"] = mean_watermarked_spike_entropy
    evaluation_results["mean_unwatermarked_spike_entropy"] = mean_unwatermarked_spike_entropy
    evaluation_results["list_watermarked_shannon_entropy"] = watermarked_shannon_entropies
    evaluation_results["list_unwatermarked_shannon_entropy"] = unwatermarked_shannon_entropies
    evaluation_results["list_watermarked_spike_entropy"] = watermarked_spike_entropies
    evaluation_results["list_unwatermarked_spike_entropy"] = unwatermarked_spike_entropies

    if dataset == "mbpp_plus":
        evaluation_results_file_name = f"evaluator/evalplus/{dataset}_{llm}_{watermark_algorithm}_{seed}_{temperature}_detection_results.pkl"
        with open(evaluation_results_file_name, 'wb') as f:
            pickle.dump(evaluation_results, f)
        evaluation_results_file_name = f"evaluator/evalplus/{dataset}_{llm}_{watermark_algorithm}_{seed}_{temperature}_detection_results.jsonl"
        del evaluation_results["list_watermarked_shannon_entropy"]
        del evaluation_results["list_unwatermarked_shannon_entropy"]
        del evaluation_results["list_watermarked_spike_entropy"]
        del evaluation_results["list_unwatermarked_spike_entropy"]
        with open(evaluation_results_file_name, 'w') as f:
            for key, value in evaluation_results.items():
                f.write(f"{key}: {value}\n")
    else: 
        evaluation_results_file_name = f"evaluator/{dataset}/{dataset}_{llm}_{watermark_algorithm}_{seed}_{temperature}_detection_results.pkl"
        with open(evaluation_results_file_name, 'wb') as f:
            pickle.dump(evaluation_results, f)
        evaluation_results_file_name = f"evaluator/{dataset}/{dataset}_{llm}_{watermark_algorithm}_{seed}_{temperature}_detection_results.jsonl"
        del evaluation_results["list_watermarked_shannon_entropy"]
        del evaluation_results["list_unwatermarked_shannon_entropy"]
        del evaluation_results["list_watermarked_spike_entropy"]
        del evaluation_results["list_unwatermarked_spike_entropy"]
        with open(evaluation_results_file_name, 'w') as f:
            for key, value in evaluation_results.items():
                f.write(f"{key}: {value}\n")