import os
import time

def read_csv():
    # get cwd
    cwd = os.getcwd()
    csv_file = os.path.join(cwd, 'cve.csv')
    with open(csv_file, 'r') as f:
        lines = f.readlines()

    return lines

def extract_func(cve):
    '''python3 /root/asm2vec-pytorch/scripts/bin2asm.py -i /root/elf/CVE_2018_0735_pre -o /root/asm2vec-pytorch/data/CVE_2018_0735_pre'''
    # To string
    bin2asm = os.join(os.getcwd(), 'scripts/bin2asm.py')
    elf_path = os.join(os.getcwd(), 'elf', cve)
    output_path = os.join(os.getcwd(), 'data', cve)
    cmd = f'python3 {bin2asm} -i {elf_path} -o {output_path}'
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