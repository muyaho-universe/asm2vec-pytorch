# This code conduct 5-fold cross validation on the given dataset
# The dataset is consisted like this:
'''
current_dir - binaries - CVE_XXXX_XXXX_pre_####
                         ...
                         CVE_XXXX_XXXX_post_####                
                         CVE_YYYY_YYYY_pre_####
                         ...
                         CVE_YYYY_YYYY_post_####
                         ...
'''
# The dataset is divided into 5 folds, and each fold is consisted of 4/5 of the dataset
# The model is trained on 4/5 of the dataset and tested on the remaining 1/5 of the dataset
# For example, if there are 10 CVEs and each CVE's pre and post version has 100 functions for each, then the first fold will use 10 CVEs' pre and post version's 80 functions for each as a training set and the remaining 20 functions for each as a test set.
# The second fold will use the first fold's test set as a training set and the first fold's training set as a test set.
# This process will be repeated until the fifth fold.
# For each fold, the model will be trained on the training set and tested on the test set.
# For testing, the model is used to calculate the cosine similarity between the target binary which is in the test set and the standard binary which is compiled by the standard compiler options; O0, O1, O2, O3, Os, and Ofast.
# The stardard binary's format is like this: CVE_XXXX_XXXX_pre_O0, CVE_XXXX_XXXX_pre_O1, CVE_XXXX_XXXX_pre_O2, CVE_XXXX_XXXX_pre_O3,  CVE_XXXX_XXXX_post_O0, CVE_XXXX_XXXX_post_O1, CVE_XXXX_XXXX_post_O2, CVE_XXXX_XXXX_post_O3
# The cosine similarity is calculated between the target binary and the standard binary.
# For each standard optimization level, the cosine similarity is calculated and the target binary will be classified as pre or post based on the average of cosine similarity.
# For example, if the cosine similarity between the target binary, whose name is CVE_XXXX_XXXX_pre_####, and the standard binary, whose name would be CVE_XXXX_XXXX_pre_O0-3, and CVE_XXXX_XXXX_post_O0-3, is like this: avg(pre_O0-3): 0.5, avg(post_O0-3): 0.4 -> 'target binary is pre'
# The ground truth is the target binary's actual version, pre or post. This will be in the its name.
# For example, if the target binary's name is CVE_XXXX_XXXX_pre_####, then the ground truth is 'pre'.
# The accuracy, precision, recall, and f1-score will be calculated for each fold.
# The final result will be the average of accuracy, precision, recall, and f1-score for each fold.

import torch
import torch.nn as nn
import click
import asm2vec
import os
import random
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold
import multiprocessing as mp
import hashlib

def cosine_similarity(v1, v2):
    return (v1 @ v2 / (v1.norm() * v2.norm())).item()

def get_cve(file):
    return file.split('_')[0] + '_' + file.split('_')[1] + '_' + file.split('_')[2]

def process_fold(train_files, test_files, std_cve, gt):
    suffix = hashlib.md5(' '.join(test_files).encode()).hexdigest()[:8]
    model = 'model_' + suffix + '.pt'
    train_model(train_files, model)
    fold_confusion_matrix = np.zeros((2, 2))
    y_true = []
    y_scores = []

    for test_file in test_files:
        cve = get_cve(test_file)
        score = {'pre': 0, 'post': 0}
        for patch in ['pre', 'post']:
            for std_file in std_cve[cve][patch]:
                similarity = compare_function(test_file, std_file, model)
                score[patch] += similarity
            score[patch] /= len(std_cve[cve][patch])

        # Calculate probabilities based on similarity scores
        y_true.append(1 if gt[test_file] == 'pre' else 0)  # 1 for 'pre', 0 for 'post'
        y_scores.append(score['pre'] - score['post'])  # Difference as a score for AUC calculation
        # Predict based on which score is higher
        prediction = 'pre' if score['pre'] > score['post'] else 'post'

        if gt[test_file] == 'pre':
            if prediction == 'pre':
                fold_confusion_matrix[0][0] += 1
            else:
                fold_confusion_matrix[0][1] += 1
        else:
            if prediction == 'post':
                fold_confusion_matrix[1][1] += 1
            else:
                fold_confusion_matrix[1][0] += 1
    # Calculate AUC for this fold
    fold_auc = roc_auc_score(y_true, y_scores)
    return fold_confusion_matrix, fold_auc
# don't need to follow compare.py

def five_fold_cross_validation(bin_path, std_cve, gt):
    '''
    new_i = ['CVE_2018_0735_post_839ead94_ec_scalar_mul_ladder', 'CVE_2018_0735_pre_dad21840_ec_scalar_mul_ladder', 'CVE_2018_0735_post_839f15dd_ec_scalar_mul_ladder']
    for i in range(len(new_i)):
        new_i[i] =  ipath + '/' + new_i[i]
    Don't need to use bin_path itself. 
    '''
    
    files = os.listdir(bin_path)
    fold_size = len(files) // 5
    kf = KFold(n_splits=5)
    confusion_matrix_total = np.zeros((2, 2))
    auc_scores = []

    fold_args = []
    for train_index, test_index in kf.split(files):
        train_files = [files[i] for i in train_index]
        test_files = [files[i] for i in test_index]
        train_files = [os.path.join(bin_path, file) for file in train_files]
        test_files = [os.path.join(bin_path, file) for file in test_files]
        fold_args.append((train_files, test_files, std_cve, gt))
        
    # Use multiprocessing to speed up the process
    with mp.Pool(mp.cpu_count()) as pool:
        fold_confusion_matrices  = pool.starmap(process_fold, fold_args)
    
    for fold_confusion_matrix in fold_confusion_matrices:
        confusion_matrix_total += fold_confusion_matrix
    
    accuracy = (confusion_matrix_total[0][0] + confusion_matrix_total[1][1]) / np.sum(confusion_matrix_total)
    precision = confusion_matrix_total[0][0] / (confusion_matrix_total[0][0] + confusion_matrix_total[0][1]) if (confusion_matrix_total[0][0] + confusion_matrix_total[0][1]) != 0 else 0
    recall = confusion_matrix_total[0][0] / (confusion_matrix_total[0][0] + confusion_matrix_total[1][0]) if (confusion_matrix_total[0][0] + confusion_matrix_total[1][0]) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    avg_auc = np.mean(auc_scores)  # Calculate the average AUC across all folds


    return accuracy, precision, recall, f1, avg_auc, confusion_matrix_total

# This method make the ground truth for the binaries
# The ground truth is the target binary's actual version, pre or post. This will be in the its name.
# For example, if the target binary's name is CVE_XXXX_XXXX_pre_####, then the ground truth is 'pre'.
# This method walks through the 'cwd/binaries' and saves the ground truth for each binary in dictionary
def make_gt(dirs, std_dirs):
    gt = {}
    
    for root, dirs, files in os.walk(dirs):
        for file in files:
            if 'pre' in file:
                gt[file] = 'pre'
            else:
                gt[file] = 'post'
    return gt

def train_model(ipath, opath):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = None
    functions, tokens = asm2vec.utils.load_data(ipath)

    def callback(context):
        progress = f'{context["epoch"]} | time = {context["time"]:.2f}, loss = {context["loss"]:.4f}'
        if context["accuracy"]:
            progress += f', accuracy = {context["accuracy"]:.4f}'
        # print(progress)
        asm2vec.utils.save_model(opath, context["model"], context["tokens"])

    model = asm2vec.utils.train(
        functions,
        tokens,
        model=model,
        embedding_size=200,
        batch_size=1024,
        epochs=100,
        neg_sample_num=25,
        calc_acc=True,
        device=device,
        mode = 'train',
        callback=callback,
        learning_rate=0.025
    )

def compare_function(standard, target, mpath):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokens = asm2vec.utils.load_model('model.pt', device=device)
    functions, tokens_new = asm2vec.utils.load_data([standard, target])
    tokens.update(tokens_new)
    model.update(2, tokens.size())
    model = model.to(device)
    model = asm2vec.utils.train(
        functions,
        tokens,
        model=model,
        embedding_size=200,
        batch_size=1024,
        epochs=10,
        neg_sample_num=25,
        device=device,
        mode='test',
        learning_rate=0.025
    )
    v1, v2 = model.to('cpu').embeddings_f(torch.tensor([0, 1]))
    return cosine_similarity(v1, v2)

def get_std_cve_files(std_path):
    files = os.listdir(std_path)
    std_files = {}
    for file in files:
        cve = get_cve(file)
        if cve not in std_files:
            std_files[cve] = {'pre': [], 'post': []}
        file = os.path.join(std_path, file)
        if 'pre' in file:
            std_files[cve]['pre'].append(file)
        else:
            std_files[cve]['post'].append(file)
    return std_files

@click.command()
@click.option('-i', '--input', 'ipath', help='training data folder', required=True)
@click.option('-o', '--output', 'opath', default='model.pt', help='output model path', show_default=True)
def main(ipath, opath):
    bin_path = os.path.join(os.getcwd(), ipath)
    std_path = os.path.join(os.getcwd(), 'std_elf')
    output_path= os.path.join(os.getcwd(), opath)
    gt = make_gt(bin_path, std_path)
    std_cve = get_std_cve_files(std_path)
    acc, prec, recall, f1, avg_auc, confusion_matrix = five_fold_cross_validation(bin_path, std_cve, gt)

    # 결과를 텍스트 파일로 저장
    with open("cross_validation_results.txt", "w") as f:
        f.write(f"accuracy: {acc}\n")
        f.write(f"precision: {prec}\n")
        f.write(f"recall: {recall}\n")
        f.write(f"f1: {f1}\n")
        f.write(f"avg_auc: {avg_auc}\n")
        f.write(f"confusion_matrix: \n{confusion_matrix}\n")

    print(f'accuracy: {acc}, precision: {prec}, recall: {recall}, f1: {f1}, avg_auc: {avg_auc}, confusion_matrix: {confusion_matrix}')

if __name__ == '__main__':
    main()
    