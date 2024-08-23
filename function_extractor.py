import os
import time

def read_csv():
    csv_file = '/root/asm.csv'
    with open(csv_file, 'r') as f:
        lines = f.readlines()

    return lines

def extract_func(cve):
    '''python3 /root/asm2vec-pytorch/scripts/bin2asm.py -i /root/elf/CVE_2018_0735_pre -o /root/asm2vec-pytorch/data/CVE_2018_0735_pre'''
    cmd = f'python3 /root/asm2vec-pytorch/scripts/bin2asm.py -i /root/elf/{cve} -o /root/asm2vec-pytorch/data/{cve}'
    os.system(cmd)


def main():
    lines = read_csv()
    for line in lines:
        cve = line.split(',')[1]
        cve_pre = cve + '_pre'
        cve_post = cve + '_post'
        extract_func(cve_pre)
        extract_func(cve_post)
        print(cve)

if __name__ == '__main__':
    main()