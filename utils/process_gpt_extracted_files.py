#!/usr/bin/env python3

import os
import sys
import re
import json
from optparse import OptionParser

def encsort(line):
    #print(line)
    return int(line.split(": ")[0])

def gen_layer_params(group, ptype, min_layer_idx, new_content,\
                     is_attn = False, is_out_proj = False):
    is_bias = ptype == 'biases'
    finfo = group[ptype]
    pth = finfo['path']
    f = open(pth, 'r')
    if f is None:
        print(f"ERROR: could not open '{pth}' for reading",file=sys.stderr)
        exit(1)
    lines = f.read().split("\n")
    f.close()
    l0 = lines[0]
    if not is_out_proj:
        if is_bias:
            match = re.match(r"--- Layer\[(\d+)\] Biases: (\d+) ---", l0)
            if match is None or match[1] is None or match[2] is None:
                print(f"ERROR: invalid line 0 in '{pth}'", file=sys.stderr)
                exit(1)
            lidx = min_layer_idx + int(match[1])
            bcount = int(match[2])
            l0 = f"--- Layer[{lidx}] Biases: {bcount} ---"
        else:
            match = re.match(r"--- Layer\[(\d+)\] Weights: (\d+,\d+) ---", l0)
            if match is None or match[1] is None or match[2] is None:
                print(f"ERROR: invalid line 0 in '{pth}'", file=sys.stderr)
                exit(1)
            lidx = min_layer_idx + int(match[1])
            wtypes,wcount = [int(n) for n in match[2].split(',')]
            l0 = f"--- Layer[{lidx}] Weights: {wtypes},{wcount} ---"
        new_content.append(l0)
    elif is_bias and is_out_proj:
        match = re.match(r"--- Layer\[\d+\] Biases: (\d+) ---", l0)
        if match is None or match[1] is None:
            print(f"ERROR: invalid line 0 in '{pth}'", file=sys.stderr)
            exit(1)
        bcount = int(match[1])
        l0 = new_content[0]
        match = re.match(r"--- Layer\[(\d+)\] Biases: (\d+) ---", l0)
        if match is None or match[1] is None or match[2] is None:
            print(f"ERROR: invalid line 0 in new content, file=sys.stderr")
            print(l0, file=sys.stderr)
            exit(1)
        bcount += (int(match[2]) + 1) 
        lidx = int(match[1])
        l0 = f"--- Layer[{lidx}] Biases: {bcount} ---"
        new_content[0] = l0
    elif not is_bias and is_out_proj:
        lw = None
        wmatch = None
        for l in lines:
            wmatch = re.match(r"--- Layer\[\d+\] Weights: \d+,(\d+) ---", l)
            if wmatch is not None:
                lw = l
                break
        if wmatch is None or wmatch[1] is None:
            print(f"ERROR: could not find weight header line in "+
                  "'{pth}'", file=sys.stderr)
            exit(1)
        wcount = int(wmatch[1])
        wmatch = None
        lw = None
        lwidx = None
        li = 0
        for l in new_content:
            wmatch = re.match(r"--- Layer\[(\d+)\] Weights: \d+,(\d+) ---", l)
            if wmatch is not None:
                lwidx = li
                lw = l
                break
            li += 1
        if wmatch is None or wmatch[1] is None or wmatch[2] is None:
            print(f"ERROR: could not find weight header line in "+
                  "new content (path = {pth})", file=sys.stderr)
            exit(1)
        wcount += int(wmatch[2])
        lidx = int(wmatch[1])
        lw = f"--- Layer[{lidx}] Weights: 5,{wcount} ---"
        new_content[lwidx] = lw
    for l in lines[1:]:
        if len(l.strip()) == 0: continue
        if is_attn and is_bias and is_out_proj:
            l1 = new_content[1]
            l1 = f"{l1},{l},0"
            new_content[1] = l1
        else: new_content.append(l)

usage = "Usage: %prog [OPTIONS] MODEL_DIR"
parser = OptionParser(usage = usage)
parser.add_option('-f', '--force', help='Generate files even if they exists',\
    default=False, action='store_true')
parser.add_option('--save-info',metavar='PATH', help='Save file info',\
    default=None, type='string', action='store')
(options, args) = parser.parse_args()

if len(args) < 1:
    parser.print_help()
    exit(1)
model_dir = args[0]
if len(args) > 1:
    output_dir = args[1]
if not os.path.exists(model_dir):
    print("model directory not found", file = sys.stderr)
    exit(1)
if not os.path.isdir(model_dir):
    print("model directory is not a valid directory", file = sys.stderr)
    exit(1)

dirname = os.path.basename(model_dir)
if dirname != 'extracted': model_dir = os.path.join(model_dir, 'extracted')
if not os.path.exists(model_dir):
    print(f"{model_dir} not found", file = sys.stderr)
    exit(1)
if not os.path.isdir(model_dir):
    print(f"{model_dir} is not a valid directory", file = sys.stderr)
    exit(1)

max_block_idx = 0
files = []
grouped_files = {}
for fname in os.listdir(model_dir):
    fpath = os.path.join(model_dir, fname)
    if not os.path.isfile(fpath): continue
    if fpath.endswith('.bin'): continue
    #print(fname)
    finfo = {'name': fname, 'path': fpath}
    block_idx = None
    if fname.startswith("h"):
        match = re.match(r"h([0-9]+)\.(.*)", fname) 
        if match is not None:
            #print(match)
            block_idx = match[1]
            #print(block_idx)
            if block_idx is not None:
                block_idx = int(block_idx)
                finfo['block'] = block_idx
                if block_idx > max_block_idx: max_block_idx = block_idx
    if fname.endswith('.b'):
        finfo['type'] = 'biases'
    elif fname.endswith('.w') or fname.endswith('.g'): finfo['type'] = 'weights'
    if 'type' in finfo:
        ptype = finfo['type']
        fcomps = fname.split('.')
        stem = '.'.join(fcomps[:-1])
        is_attn = fcomps[1] == 'attn'
        is_out_proj = False
        if is_attn:
            stem = '.'.join(fcomps[0:2]) 
            is_out_proj = fcomps[2] == 'c_proj'
        group = {}
        if stem in grouped_files: group = grouped_files[stem]
        else: grouped_files[stem] = group
        info = group
        if is_out_proj:
            if not 'out_proj' in group: group['out_proj'] = {}
            info = group['out_proj']
        group['is_attn'] = is_attn
        info[ptype] = finfo
        if block_idx is not None: group['block_idx'] = block_idx
        finfo['group'] = stem
        
    files.append(finfo)

if len(files) == 0:
    print(f"model directory is empty!")
    exit(1)
if max_block_idx == 0:
    print(f"could not find files for transformer block")
    exit(1)

block_size = 7
n_blocks = max_block_idx + 1
print(f"Blocks count: {n_blocks}")
#print(grouped_files)
if options.save_info:
    with open(options.save_info, 'w') as f:
        f.write(json.dumps(grouped_files, indent=4))
        f.close()

destdir = os.path.join(model_dir, 'processed')
if not os.path.exists(destdir): os.mkdir(destdir)
for name in grouped_files.keys():
    group = grouped_files[name]
    outpath = os.path.join(destdir, name)
    if not options.force and os.path.exists(outpath): continue
    print(f"Generating {name}")
    new_content = []
    is_attn = ('is_attn' in group and group['is_attn'] is True)
    block_idx = None
    if 'block_idx' in group: block_idx = group['block_idx']
    min_layer_idx = 3
    if block_idx is not None:
        min_layer_idx += (block_idx * block_size) 
    else: min_layer_idx += (n_blocks * block_size)
    for ptype in ['biases', 'weights']:
        if not ptype in group:
            print(f"ERROR: missing '{ptype}' in group {name}", file=sys.stderr)
            exit(1)
        gen_layer_params(group, ptype, min_layer_idx, new_content,\
            is_attn = is_attn)
    if is_attn:
        if not 'out_proj' in group:
            print(f"ERROR: missing 'out_proj' in group {name}", file=sys.stderr)
            exit(1)
        out_proj = group['out_proj']
        for ptype in ['biases', 'weights']:
            if not ptype in out_proj:
                print(f"ERROR: missing '{ptype}' in group {name} 'out_proj'",\
                    file=sys.stderr)
                exit(1)
            gen_layer_params(out_proj, ptype, min_layer_idx, new_content,\
                             is_attn = True, is_out_proj = True)
        new_content.append('---')
    f = open(outpath, 'w')
    if f is None:
        print(f"ERROR: could not open '{outpath}' for writing",file=sys.stderr)
        exit(1)
    f.write("\n".join(new_content) + "\n")
    f.close()

for finfo in files:
    grouped = False
    if 'group' in finfo: grouped = (finfo['group'] is not None)
    if not grouped and 'path' in finfo:
        #print(finfo['path'])
        f = open(finfo['path'], 'r')
        if not f:
            print(f"ERROR: could not open '{finfo['path']}")
            exit(1)
        lines = f.read().split("\n")
        f.close()
        l0 = lines[0]
        if not l0: continue
        match = re.match(r"--- Layer\[(\d+)\]\s+([a-zA-Z]+)", l0)
        if not match: continue
        if match[1] is not None and match[2] == 'Biases': continue
        print(f"Generating biases header for: {finfo['path']}")
        lidx = int(match[1])  
        l0 = f"--- Layer[{lidx}] Biases: 0 ---"
        lines.insert(0, l0)
        new_content = "\n".join(lines)
        f = open(finfo['path'], 'w')
        if not f:
            print(f"ERROR: could not open '{finfo['path']} for writing",\
                  file=sys.stderr)
            exit(1)
        f.write(new_content)
        f.close()

encoder = os.path.join(os.path.dirname(model_dir), 'encoder.json')
ps_encoder = os.path.join(os.path.dirname(model_dir), 'psyc_encoder.txt')
if not os.path.exists(ps_encoder):
    if not os.path.exists(encoder):
        print(f"ERROR: could not find {encoder}", file=sys.stderr)
        exit(1)
    print(f"Generating psyc_encoder.txt")
    f = open(encoder, 'r')
    if f is None:
        print(f"ERROR: could not open '{encoder}'", file=sys.stderr)
    enc = f.read()
    f.close()
    enc = json.loads(enc)
    f = open(ps_encoder, 'w')
    if f is None:
        print(f"ERROR: could not open '{ps_encoder}' for writing",\
              file=sys.stderr)
    last_index = -1
    do_sort = False
    lines = []
    for k in enc.keys():
        index = enc[k]
        if not do_sort:
            do_sort = (index - last_index) > 1
        last_index = index
        #f.write(f"{index}: {k}\n")
        lines.append(f"{index}: {k}")

    if do_sort:
        lines = sorted(lines, key = encsort)
    #print("\n".join(lines[0:10]))
    f.write("\n".join(lines))
    f.close()

