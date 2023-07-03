#!/usr/bin/env python3

import sys
import os
import re
import json
import numpy as np
import tensorflow as tf
from optparse import OptionParser
config = {}

def removeprefix(s, prefix):
    if 'removeprefix' in dir(s): return s.removeprefix(prefix)
    idx = None
    try:
        idx = s.index(prefix)
    except ValueError:
        idx = None
    if idx is None or idx != 0: return s
    return s.replace(prefix, '', 1)

def get_config(var):
    global config
    var = removeprefix(var, "model/")
    if var in config: return config[var]
    cfg = None
    for cfgvar in config.keys():
        c = config[cfgvar]
        is_exp = ('is_exp' in c and c['is_exp'] is True)
        if is_exp:
            match = re.match(cfgvar, var)    
            if match is not None:
                cfg = c
                break
    return cfg

def set_config(var, key, val):
    global config
    var = removeprefix(var, "model/")
    is_exp = (var.find('*') >= 0)
    if is_exp:
        var = var.replace('*', '.*')
    if not var in config: config[var] = {}
    config[var]['is_exp'] = is_exp
    config[var][key] = val
    return val

def save_var(array, fpath, cfg, options):
    array = np.squeeze(array)
    save_transposed_to = None
    if 'save_transposed_to' in cfg:
        save_transposed_to = cfg['save_transposed_to']
    if not options.force and os.path.exists(fpath):
        if save_transposed_to is None or os.path.exists(save_transposed_to):
            return
    print(f"Extracting: {varname}")
    split = 0
    if 'split' in cfg: split = int(cfg['split'])
    f = open(fpath, 'w')
    if not f:
        print(f"FATAL: could not open '{fpath}' for writing", file=sys.stderr)
        exit(1)
    transpose = ('transpose' in cfg and cfg['transpose'] is True)
    if transpose:
        if save_transposed_to is not None:
            if not 'T' in cfg:
                cfg['T'] = True
                dest = cfg['save_transposed_to']
                if not dest.startswith("/"):
                    dest = os.path.join(os.path.dirname(fpath), dest)
                save_var(array.T, dest, cfg, options)
                cfg.pop('T', None)
        else:
            if split < 2: array = array.T
    omit_shape = options.omit_shape is True
    psyc_param = None
    if 'psyc_param' in cfg:
        psyc_param = cfg['psyc_param']
        omit_shape = True
        size_str = ''
        ptype = psyc_param['type']
        if ptype == 'weights':
            size_str = f"{psyc_param['wcount']},{array.size}"
        else: size_str = f"{array.size}"
        lidx = psyc_param['lidx']
        if 'T' in cfg and cfg['T'] is True and 'transposed_lidx' in psyc_param:
            lidx = psyc_param['transposed_lidx']
        f.write(f"--- Layer[{lidx}] {ptype.capitalize()}: {size_str} ---\n") 
    elif not options.omit_shape:
        shapestr = ','.join([str(d) for d in array.shape])
        f.write(f"shape:{shapestr}\n") 
    arrays = [array]
    if split > 1:
        arrays = np.split(array, split, axis=-1)
        if transpose:
            arrays = [ar.T for ar in arrays]
    for ar in arrays:
        ar = ar.ravel()
        f.write(','.join([str(n) for n in ar]) + "\n")
    f.close()

output_dir = None
usage = "Usage: %prog [OPTIONS] MODEL_DIR [OUTPUT_DIR]"
parser = OptionParser(usage = usage)
parser.add_option('-c', '--config', metavar='PATH',\
    help='Load config from file', action='store', type = 'string',
    default = None)
parser.add_option('','--omit-shape', help='Omit shape in saved files',\
    default=False, action='store_true')
parser.add_option('-t', '--transpose', metavar='VARNAME',action='append',\
    type='string', default=None, help='Transpose matrix')
parser.add_option('--save-transposed-to', metavar='VARNAME PATH',\
    action='append', type='string', default=None, nargs = 2,\
    help='Save transposed matrix to separate file')
parser.add_option('-s','--split', metavar='VARNAME SLICES',action='append',\
    type='string', default=None, nargs = 2, help='Split matrix')
parser.add_option('-p','--psyc-param', metavar='VAR PARAM LAYER_IDX WCOUNT',\
    action='append', nargs = 4, default=None,\
    help='Write matrix as PsyC paramaters (weights|biases)')
parser.add_option('-f', '--force', help='Generate files even if they exists',\
    default=False, action='store_true')
parser.add_option('--save-config',metavar='PATH', help='Save configuration',\
    default=None, type='string', action='store')
#parser.add_option('-v', '--verbose',action="store_true")
#parser.add_option('--vv',action="store_true")
(options, args) = parser.parse_args()

if options.config is not None:
    if not os.path.exists(options.config):
        print(f"config file not found", file = sys.stderr)
        exit(1)
    f = open(options.config)
    if f is None:
        print(f"could not open config file {options.config}", file = sys.stderr)
        exit(1)
    config = json.loads(f.read())
    f.close()

if options.transpose is not None:
    for vname in options.transpose:
        set_config(vname, 'transpose', True)
if options.save_transposed_to is not None:
    for vname, fpath in options.save_transposed_to:
        set_config(vname, 'save_transposed_to', fpath)
if options.psyc_param is not None:
    for vname,psparam,lidx,wcount in options.psyc_param:
        if psparam != 'biases' and psparam != 'weights':
            print(f"FATAL: invalid psyc param {psparam}. " +
                  "Allowed: weights|biases", file = sys.stderr)
            exit(1)
        par = {
            'type': psparam,
            'lidx': lidx,
            'wcount': wcount
        }
        set_config(vname, 'psyc_param', par)
if options.split is not None:
    for vname,slices in options.split:
        slices = int(slices)
        set_config(vname, 'split', slices)
if options.save_config is not None:
    f = open(options.save_config, 'w')
    if f is None:
        print(f"could not open {options.save_config} for writing",\
              file = sys.stderr)
        exit(1)
    f.write(json.dumps(config, indent=4))
    f.close()

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

if output_dir is None:
    output_dir = os.path.join(model_dir, 'extracted')
    if not os.path.exists(output_dir): os.mkdir(output_dir)

ckpt_path = tf.train.latest_checkpoint(model_dir)
if ckpt_path is None:
    print("invalid tensorflow model directory", file = sys.stderr)
    exit(1)

init_vars = tf.train.list_variables(ckpt_path)
for name, _ in init_vars:
    #print(name)
    array = np.squeeze(tf.train.load_variable(ckpt_path, name))
    varname = name
    name = removeprefix(name, "model/")
    cfg = get_config(name)
    if cfg is None: cfg = {}
    fname = name.replace('/', '.')
    fpath = os.path.join(output_dir, fname)
    save_var(array, fpath, cfg, options)
print(f"DONE: model extracted into '{output_dir}'")
