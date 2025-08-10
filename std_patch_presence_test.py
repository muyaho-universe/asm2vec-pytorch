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
    
    results = {}
    correct = 0
    total = 0
    score_sum = 0.0
    
    with open('std_postpre_results.txt', 'w') as f:
        f.write('target\tpredicted_postpre\ttrue_postpre\tscore_same\tscore_diff\tcorrect\n')
        
        for cve in tqdm(cve_dict.keys(), desc='Processing CVEs'):
            # 해당 CVE가 post와 pre를 모두 가지고 있는지 확인
            if 'post' not in cve_dict[cve] or 'pre' not in cve_dict[cve]:
                continue
                
            # post와 pre 각각에 대해 처리
            for postpre in ['post', 'pre']:
                opts = list(cve_dict[cve][postpre].keys())
                if len(opts) == 0:
                    continue
                    
                # 랜덤하게 하나의 옵션 선택
                opt = random.choice(opts)
                target = cve_dict[cve][postpre][opt]
                target_postpre = postpre
                
                # 같은 post/pre 내에서 다른 옵션 선택 (자기 자신 제외)
                same_postpre_opts = [o for o in opts if o != opt]
                if not same_postpre_opts:
                    # 같은 post/pre 내에 다른 옵션이 없으면 건너뛰기
                    continue
                same_postpre_opt = random.choice(same_postpre_opts)
                same_postpre_file = cve_dict[cve][postpre][same_postpre_opt]
                
                # 다른 post/pre에서 랜덤 옵션 선택
                other_postpre = 'pre' if postpre == 'post' else 'post'
                other_postpre_opts = list(cve_dict[cve][other_postpre].keys())
                if not other_postpre_opts:
                    continue
                other_postpre_opt = random.choice(other_postpre_opts)
                other_postpre_file = cve_dict[cve][other_postpre][other_postpre_opt]
                
                # 같은 post/pre와 2번 비교하여 평균 계산
                same_scores = []
                for _ in range(2):
                    score = compare_function(target, same_postpre_file, model_path)
                    same_scores.append(score)
                avg_same_score = sum(same_scores) / len(same_scores)
                
                # 다른 post/pre와 2번 비교하여 평균 계산
                diff_scores = []
                for _ in range(2):
                    score = compare_function(target, other_postpre_file, model_path)
                    diff_scores.append(score)
                avg_diff_score = sum(diff_scores) / len(diff_scores)
                
                # 예측: 더 높은 평균 점수를 가진 쪽으로 판단
                if avg_same_score > avg_diff_score:
                    predicted_postpre = target_postpre
                    best_score = avg_same_score
                else:
                    predicted_postpre = other_postpre
                    best_score = avg_diff_score
                
                # 정답 확인
                true_postpre = target_postpre
                is_correct = (predicted_postpre == true_postpre)
                
                total += 1
                if is_correct:
                    correct += 1
                score_sum += best_score
                
                print(f"{os.path.basename(target)} -> {predicted_postpre} (same: {avg_same_score:.4f}, diff: {avg_diff_score:.4f}) {'O' if is_correct else 'X'}")
                
                # 결과 기록
                f.write(f"{os.path.basename(target)}\t{predicted_postpre}\t{true_postpre}\t{avg_same_score:.4f}\t{avg_diff_score:.4f}\t{int(is_correct)}\n")
    
    accuracy = correct / total if total > 0 else 0
    avg_score = score_sum / total if total > 0 else 0
    
    # 결과 요약
    with open('std_postpre_results.txt', 'a') as f:
        f.write(f"\nTotal: {total}\nCorrect: {correct}\nAccuracy: {accuracy:.4f}\nAvg_score: {avg_score:.4f}\n")
    
    print(f"\nTotal samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average score: {avg_score:.4f}")

if __name__ == '__main__':
    main()