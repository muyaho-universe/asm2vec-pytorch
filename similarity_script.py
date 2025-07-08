import os
import torch
import asm2vec
from tqdm import tqdm
import random

def cosine_similarity(v1, v2):
    return (v1 @ v2 / (v1.norm() * v2.norm())).item()

def get_cve(file):
    file = file.split('_')
    return '_'.join(file[:3])

def get_std_cve_files(std_path):
    files = os.listdir(std_path)
    std_files = {}
    for file in files:
        cve = get_cve(file)
        if cve not in std_files:
            std_files[cve] = []
        std_files[cve].append(os.path.join(std_path, file))
    return std_files

def compare_function(target, standard, model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokens = asm2vec.utils.load_model(model_path, device=device)
    functions, tokens_new = asm2vec.utils.load_data([target, standard])
    tokens.update(tokens_new)
    model.update(2, tokens.size())
    model = model.to(device)
    model = asm2vec.utils.train(
        functions,
        tokens,
        model=model,
        embedding_size=200,
        batch_size=1024,
        epochs=30,
        neg_sample_num=25,
        device=device,
        mode='test',
        learning_rate=0.025
    )
    v1, v2 = model.to('cpu').embeddings_f(torch.tensor([0, 1]))
    return cosine_similarity(v1, v2)

def extract_cve_name(filename):
    # 예: CVE_2022_0778_pre3_7cc08da3 -> CVE_2022_0778
    parts = filename.split('_')
    return '_'.join(parts[:3])

def main():
    bin_path = os.path.join(os.getcwd(), 'binaries')
    std_path = os.path.join(os.getcwd(), 'std_elf')
    model_path = os.path.join(os.getcwd(), 'model_5988db9d.pt')  # 기존 학습된 모델 사용

    std_cve = get_std_cve_files(std_path)
    all_bin_files = [os.path.join(bin_path, f) for f in os.listdir(bin_path)]
    # 10000개 샘플링
    bin_files = random.sample(all_bin_files, min(1000, len(all_bin_files)))

    # 각 바이너리의 ground truth CVE 추출
    gt_cves = set(extract_cve_name(os.path.basename(bin_file)) for bin_file in bin_files)
    all_cves = list(std_cve.keys())
    # ground truth CVE를 무조건 포함하여 30개 샘플링
    remaining_cves = list(set(all_cves) - gt_cves)
    sampled_cves = list(gt_cves)
    if len(sampled_cves) < 30:
        sampled_cves += random.sample(remaining_cves, min(30 - len(sampled_cves), len(remaining_cves)))
    else:
        sampled_cves = random.sample(sampled_cves, 30)
    sampled_std_cve = {cve: std_cve[cve] for cve in sampled_cves}

    results = {}
    correct = 0
    total = 0
    score_sum = 0.0
    with open('similarity_results.txt', 'w') as f:
        f.write('binary\tpredicted_cve\ttrue_cve\tscore\tcorrect\n')
        for bin_file in tqdm(bin_files, desc='Processing binaries'):
            cve_scores = {}
            for cve, std_files in sampled_std_cve.items():
                std_files_o = [f for f in std_files if any(opt in f for opt in ['_O0', '_O1', '_O2', '_O3'])]
                if not std_files_o:
                    continue
                sims = []
                for std_file in std_files_o:
                    sim = compare_function(bin_file, std_file, model_path)
                    sims.append(sim)
                if sims:
                    cve_scores[cve] = sum(sims) / len(sims)
            if cve_scores:
                best_cve = max(cve_scores, key=cve_scores.get)
                score = cve_scores[best_cve]
                bin_name = os.path.basename(bin_file)
                true_cve = extract_cve_name(bin_name)
                is_correct = (true_cve == best_cve)
                results[bin_name] = best_cve
                total += 1
                if is_correct:
                    correct += 1
                score_sum += score
                print(f"{bin_name} -> {best_cve} (score: {score:.4f}) {'O' if is_correct else 'X'}")
                f.write(f"{bin_name}\t{best_cve}\t{true_cve}\t{score:.4f}\t{int(is_correct)}\n")
    accuracy = correct / total if total > 0 else 0
    avg_score = score_sum / total if total > 0 else 0
    with open('similarity_results.txt', 'a') as f:
        f.write(f"\nTotal: {total}\nCorrect: {correct}\nAccuracy: {accuracy:.4f}\nAvg_score: {avg_score:.4f}\n")

if __name__ == '__main__':
    main()
