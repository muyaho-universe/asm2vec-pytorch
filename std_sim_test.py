import os
import torch
import asm2vec
from tqdm import tqdm
import random

def cosine_similarity(v1, v2):
    return (v1 @ v2 / (v1.norm() * v2.norm())).item()

def parse_std_elf_files(std_path):
    files = os.listdir(std_path)
    cve_dict = {}
    for file in files:
        parts = file.split('_')
        if len(parts) < 5:
            continue
        cve = '_'.join(parts[:3])
        postpre = parts[3]  # 'post' or 'pre'
        opt = parts[4]      # 'O0'~'O3'
        cve_dict.setdefault(cve, {}).setdefault(postpre, {})[opt] = os.path.join(std_path, file)
    return cve_dict

def compare_function(target, standard, model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokens = asm2vec.utils.load_model(model_path, device=device)
    functions, tokens_new = asm2vec.utils.load_data([target, standard])
    tokens.update(tokens_new)
    model.update(2, tokens.size())
    model = model.to(device)
    # NOTE: 학습 파라미터 조정
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

def main():
    std_path = os.path.join(os.getcwd(), 'std_elf')
    model_path = os.path.join(os.getcwd(), 'model_5988db9d.pt')
    cve_dict = parse_std_elf_files(std_path)
    cve_list = list(cve_dict.keys())
    postpre_list = ['post', 'pre']
    opt_list = ['O0', 'O1', 'O2', 'O3']
    results = {}
    correct = 0
    total = 0
    score_sum = 0.0
    with open('std_similarity_results.txt', 'w') as f:
        f.write('target\tpredicted_cve\ttrue_cve\tscore\tcorrect\n')
        for cve in tqdm(cve_list, desc='Processing CVEs'):
            postpre_choices = [pp for pp in postpre_list if pp in cve_dict[cve]]
            if not postpre_choices:
                continue
            postpre = random.choice(postpre_choices)
            opts = list(cve_dict[cve][postpre].keys())
            if len(opts) < 2:
                continue
            opt = random.choice(opts)
            target = cve_dict[cve][postpre][opt]
            other_opts = [o for o in opts if o != opt]
            if not other_opts:
                continue
            std1_opt = random.choice(other_opts)
            std1 = cve_dict[cve][postpre][std1_opt]
            # 나머지 CVE에서 post/pre, O0~O3 중 랜덤 1개씩 선택
            sampled_cves = [x for x in cve_list if x != cve]
            sampled_std = []
            for scve in sampled_cves:
                spp_choices = [pp for pp in postpre_list if pp in cve_dict[scve]]
                if not spp_choices:
                    continue
                spp = random.choice(spp_choices)
                sopts = list(cve_dict[scve][spp].keys())
                sopt = random.choice(sopts)
                sampled_std.append(cve_dict[scve][spp][sopt])
            # 비교 대상: [std1] + sampled_std
            cve_scores = {}
            for idx, std_file in enumerate([std1] + sampled_std):
                # NOTE: 반복 횟수 조정
                iterations = 2
                sum_sim = 0.0
                for _ in range(iterations):
                    sum_sim += compare_function(target, std_file, model_path)
                avg_sim = sum_sim / iterations
                key = cve if idx == 0 else sampled_cves[idx-1]
                cve_scores[key] = avg_sim
            # 예측
            best_cve = max(cve_scores, key=cve_scores.get)
            score = cve_scores[best_cve]
            true_cve = cve
            is_correct = (true_cve == best_cve)
            results[target] = best_cve
            total += 1
            if is_correct:
                correct += 1
            score_sum += score
            print(f"{os.path.basename(target)} -> {best_cve} (score: {score:.4f}) {'O' if is_correct else 'X'}")
            # 결과 기록
            f.write(f"{os.path.basename(target)}\t{best_cve}\t{true_cve}\t{score:.4f}\t{int(is_correct)}\n")
    accuracy = correct / total if total > 0 else 0
    avg_score = score_sum / total if total > 0 else 0
    # 결과 기록
    with open('std_similarity_results.txt', 'a') as f:
        f.write(f"\nTotal: {total}\nCorrect: {correct}\nAccuracy: {accuracy:.4f}\nAvg_score: {avg_score:.4f}\n")

if __name__ == '__main__':
    main()
