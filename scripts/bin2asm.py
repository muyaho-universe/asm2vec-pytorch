#!/usr/bin/env python3
import re
import os
import click
import r2pipe
import hashlib
from pathlib import Path
import time
from multiprocessing import Pool, Manager
from tqdm import tqdm

def sha3(data):
    return hashlib.sha3_256(data.encode()).hexdigest()

def validEXE(filename):
    magics = [bytes.fromhex('7f454c46')]
    with open(filename, 'rb') as f:
        header = f.read(4)
        return header in magics

def normalize(opcode):
    opcode = opcode.replace(' - ', ' + ')
    opcode = re.sub(r'0x[0-9a-f]+', 'CONST', opcode)
    opcode = re.sub(r'\*[0-9]', '*CONST', opcode)
    opcode = re.sub(r' [0-9]', ' CONST', opcode)
    return opcode

def fn2asm(pdf, minlen):
    # check
    if pdf is None:
        print('pdf is empty')
        return
    if pdf == {}:
        return
        #print(f'pdf:\n{pdf}')
    if len(pdf['ops']) < minlen:
        return
    if 'invalid' in [op['type'] for op in pdf['ops']]:
        return

    ops = pdf['ops']

    # set label
    labels, scope = {}, [op['offset'] for op in ops]
    assert(None not in scope)
    for i, op in enumerate(ops):
        if op.get('jump') in scope:
            labels.setdefault(op.get('jump'), i)

    # dump output
    output = ''
    for op in ops:
        # add label
        if labels.get(op.get('offset')) is not None:
            output += f'LABEL{labels[op["offset"]]}:\n'
        # add instruction
        if labels.get(op.get('jump')) is not None:
            output += f' {op["type"]} LABEL{labels[op["jump"]]}\n'
        else:
            output += f' {normalize(op["opcode"])}\n'
    # print(f'labels: {labels}')
    return output

def bin2asm(filename, opath, minlen, gt):
    # print(f'filename: {filename}')
    # check
    if not validEXE(filename):
        # print('error')
        return 0
    r = r2pipe.open(str(filename))
    # r.cmd('e bin.relocs.apply=true')

    # Suppress output
    r.cmd('e scr.color=false')
    # r.cmd('e scr.verbosity=0')

    r.cmd('aaaa')
    
    result = set()

    count = 0
    fff = str(filename).split('/')[-1].split('.')[0]
    cve = fff.split('_')[0]+'_'+fff.split('_')[1]+'_'+fff.split('_')[2]
    fc_name = gt[cve]
    # print(f'fc_name: {fc_name}')
    # print(f'cve: {cve}')

    for fn in r.cmdj('aflj'):
        # print(f'fn: {fn}')
        function_name = fn['name'].split('.')[-1]
        # print(f'function_name: {function_name}')
        if function_name != fc_name:
            continue
        
        r.cmd(f's {fn["offset"]}')
        asm = fn2asm(r.cmdj('pdfj'), minlen)
        if asm:
            uid = sha3(asm)
            if uid in result:
                continue
            else:
                result.add(uid)
            asm = f''' .name {fn["name"]}
 .offset {fn["offset"]:016x}
 .file {filename.name}
''' + asm
            file_name = ''
        
            file_name = str(fff)
            # print(file_name)
            with open(opath / file_name, 'w') as f:
                f.write(asm)
                count += 1
            print(f'[+] {filename}')
            break # don't need to extract other functions
    

    return count

def get_fc():
        cwd = os.getcwd()
        csv_file = os.path.join(cwd, 'cve.csv')
        gt = {}
        with open(csv_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                cve = line.split(',')[1]
                if cve not in gt:
                    gt[cve] = line.split(',')[3].replace('\n', '')   
        return gt

def process_directory(args):
    ipath, opath, minlen, gt, progress_queue = args
    fcount, bcount = 0, 0
    ipath = Path(ipath)
    opath = Path(opath)

    if os.path.isdir(ipath):
        for f in os.listdir(ipath):
            if not os.path.islink(ipath / f) and not os.path.isdir(ipath / f):
                fcount += bin2asm(ipath / f, opath, minlen, gt)
                bcount += 1
            progress_queue.put(1)
    elif os.path.exists(ipath):
        fcount += bin2asm(ipath, opath, minlen, gt)
        bcount += 1
        progress_queue.put(1)
    return fcount, bcount

@click.command()
@click.option('-i', '--input', 'ipath', help='input directory / file', required=True)
@click.option('-o', '--output', 'opath', default='asm', help='output directory')
@click.option('-l', '--len', 'minlen', default=10, help='ignore assembly code with instructions amount smaller than minlen')
@click.option('-p', '--processes', default=4, help='number of parallel processes to use')
def cli(ipath, opath, minlen, processes):
    '''
    Extract assembly functions from binary executables using multiple processes
    '''
    ipath = Path(ipath)
    opath = Path(opath)
    
    if not os.path.exists(opath):
        os.mkdir(opath)

    gt = get_fc()
    directories = [os.path.join(ipath, d) for d in os.listdir(ipath) if os.path.isdir(os.path.join(ipath, d))]
    
    # Manager for progress_queue
    manager = Manager()
    progress_queue = manager.Queue()
    args = [(d, opath, minlen, gt, progress_queue) for d in directories]

    # Initialize progress bar
    total_files = sum(len(os.listdir(d)) for d in directories)

    with tqdm(total=total_files, desc="Processing", unit="file") as pbar:
        with Pool(processes=processes) as pool:
            for _ in pool.imap_unordered(process_directory, args):
                pbar.update(progress_queue.get())

    total_fcount = sum(fcount for fcount, bcount in pool.map(process_directory, args))
    total_bcount = sum(bcount for fcount, bcount in pool.map(process_directory, args))
    print(f'[+] Total scan binary: {total_bcount} => Total generated assembly functions: {total_fcount}')

if __name__ == '__main__':
    cli()