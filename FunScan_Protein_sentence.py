#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import pickle as pkl
import subprocess
import argparse
import shutil
from shutil import which

parser = argparse.ArgumentParser(description="""FunScan_Protein_sentence.""")
parser.add_argument('--inputdir', help='FASTA file of contigs', default='FunScan_data/')
parser.add_argument('--threads', help='number of threads to use', type=int, default=64)
parser.add_argument('--proteins', help='FASTA file of predicted proteins (optional)')
parser.add_argument('--trainfolder', help='results file of train', type=str, default='FunScan_train_sentence/')
parser.add_argument('--evalfolder', help='results file of eval', type=str, default='FunScan_eval_sentence/')
parser.add_argument('--proteinsfolder', help='protein vocabulary', type=str, default='FunScan_pretreat/')
inputs = parser.parse_args()
input_fn = inputs.inputdir
train_fn = inputs.trainfolder
eval_fn = inputs.evalfolder
protein_fn = inputs.proteinsfolder
if not os.path.isdir(train_fn):
    os.makedirs(train_fn)
if not os.path.isdir(eval_fn):
    os.makedirs(eval_fn)
if not os.path.isdir(protein_fn):
    os.makedirs(protein_fn)
threads = inputs.threads

if inputs.proteins is None:
    prodigal = "prodigal"
    if which("pprodigal") is not None:
        print("Using parallelized prodigal...")
        prodigal = f'pprodigal -T {threads}'
    prodigal_train_cmd = f'{prodigal} -n -g 1 -p meta -i {input_fn}/train.rN.fa -a {train_fn}/train.rN.faa -f gff -d {train_fn}/train.rN.fna -T {threads}'
    prodigal_eval_cmd = f'{prodigal} -n -g 1 -p meta -i {input_fn}/eval.rN.fa -a {eval_fn}/eval.rN.faa -f gff -d {eval_fn}/eval.rN.fna -T {threads}'
    _ = subprocess.check_call(prodigal_train_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _ = subprocess.check_call(prodigal_eval_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
else:
    shutil.copyfile(inputs.proteins, f'{train_fn}/train.rN.fna')
    shutil.copyfile(inputs.proteins, f'{eval_fn}/eval.rN.fna')

try:
    transeq_train_cmd = f'transeq -sequence {train_fn}/train.rN.fna -outseq {train_fn}/train.rN.nuc.transeq -frame F -clean Y'
    transeq_eval_cmd = f'transeq -sequence {eval_fn}/eval.rN.fna -outseq {eval_fn}/eval.rN.nuc.transeq -frame F -clean Y'
    _ = subprocess.check_call(transeq_train_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _ = subprocess.check_call(transeq_eval_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

except Exception as e:
    print(f"An unexpected error occurred: {e}")

try:
    perl_train_cmd = f"perl -ne 'chomp;if(/>(\\S+)_\\d(\\s+\\#.+)/){{ $i++;if($i eq 1){{ print \">$1 $2\\n\"; }}if($i eq 4){{ print \"$seq\\n\"; $seq=\"\"; $i=0; }} }}else{{ $seq.=$_; }}END{{ print \"$seq\\n\"; }}' {train_fn}/train.rN.nuc.transeq > {train_fn}/train.rN.nuc.transeq.s"
    perl_eval_cmd = f"perl -ne 'chomp;if(/>(\\S+)_\\d(\\s+\\#.+)/){{ $i++;if($i eq 1){{ print \">$1 $2\\n\"; }}if($i eq 4){{ print \"$seq\\n\"; $seq=\"\"; $i=0; }} }}else{{ $seq.=$_; }}END{{ print \"$seq\\n\"; }}' {eval_fn}/eval.rN.nuc.transeq > {eval_fn}/eval.rN.nuc.transeq.s"
    _ = subprocess.check_call(perl_train_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _ = subprocess.check_call(perl_eval_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except Exception as e:
    print(f"An unexpected error occurred: {e}")

diamond_path = '/home/zh22tlsun/anaconda3/envs/diamond/bin/diamond'
diamond_db = f'{input_fn}/mmseq3.dmnd'
try:
    diamond_cmd = f'{diamond_path} blastp --threads {threads} --sensitive  -d {diamond_db} -q {input_fn}/train.rN.nuc.transeq.s -o {train_fn}/train.rN.nuc.tab -k 1 -e 1e-3'
    print("Running Diamond...")
    _ = subprocess.check_call(diamond_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    diamond_out_fp = f"{train_fn}/train.rN.nuc.tab"
    database_abc_fp = f"{train_fn}/train.rN.nuc.abc"
    _ = subprocess.check_call("awk '{{print $1,$2,$11}}' {0} > {1}".format(diamond_out_fp, database_abc_fp), shell=True)
except:
    print("diamond blastp failed")
    exit(1)


proteins_df = pd.read_csv(f'{protein_fn}/train_funscan_proteins')  # train_29_protein
proteins_df.dropna(axis=0, how='any', inplace=True)
pc2wordsid = {pc: idx for idx, pc in enumerate(sorted(set(proteins_df['cluster'].values)))}  # 映射簇名加簇名id
protein2pc = {protein: pc for protein, pc in
              zip(proteins_df['protein_id'].values, proteins_df['cluster'].values)}  # 蛋白质和蛋白质簇的映射
blast_df = pd.read_csv(f"{train_fn}/train.rN.nuc.abc", sep=' ', names=['query', 'ref', 'evalue'])
contig2pcs = {}
for query, ref, evalue in zip(blast_df['query'].values, blast_df['ref'].values, blast_df['evalue'].values):
    conitg = query.rsplit('_', 1)[0]
    idx = query.rsplit('_', 1)[1]
    try:
        pc = pc2wordsid[protein2pc[ref]]
    except KeyError:
        continue
    try:
        contig2pcs[conitg].append((idx, pc, evalue))
    except:
        contig2pcs[conitg] = [(idx, pc, evalue)]

# Sorted by position
from decimal import Decimal
for contig in contig2pcs:
    contig2pcs[contig] = sorted(contig2pcs[contig], key=lambda tup: int(tup[0]))
pkl.dump(contig2pcs, open(f'{train_fn}/contig2pcs.dict', 'wb'))

# Contigs2sentence
contig2id = {contig: idx for idx, contig in enumerate(contig2pcs.keys())}
id2contig = {idx: contig for idx, contig in enumerate(contig2pcs.keys())}

sentence = np.zeros((len(contig2id.keys()), 100))

sentence_weight = np.ones((len(contig2id.keys()), 100))

for row in range(sentence.shape[0]):
    contig = id2contig[row]
    pcs = contig2pcs[contig]
    for col in range(len(pcs)):
        try:
            _, sentence[row][col], sentence_weight[row][col] = pcs[col]
            sentence[row][col] += 1
        except:
            break

pkl.dump(sentence, open(f'{train_fn}/sentence.feat', 'wb'))
pkl.dump(pc2wordsid, open(f'{train_fn}/pc2wordsid.dict', 'wb'))


try:
    diamond_cmd = f'{diamond_path} blastp --threads {threads} --sensitive  -d {diamond_db} -q {input_fn}/eval.rN.nuc.transeq.s -o {eval_fn}/eval.rN.nuc.tab -k 1 -e 1e-3'
    print("Running Diamond...")
    _ = subprocess.check_call(diamond_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    diamond_out_fp = f"{eval_fn}/eval.rN.nuc.tab"
    database_abc_fp = f"{eval_fn}/eval.rN.nuc.abc"
    _ = subprocess.check_call("awk '{{print $1,$2,$11}}' {0} > {1}".format(diamond_out_fp, database_abc_fp), shell=True)
except:
    print("diamond blastp failed")
    exit(1)

blast_df = pd.read_csv(f"{eval_fn}/eval.rN.nuc.abc", sep=' ', names=['query', 'ref', 'evalue'])

contig2pcs = {}
for query, ref, evalue in zip(blast_df['query'].values, blast_df['ref'].values, blast_df['evalue'].values):

    conitg = query.rsplit('_', 1)[0]
    idx = query.rsplit('_', 1)[1]
    try:
        pc = pc2wordsid[protein2pc[ref]]
    except KeyError:

        continue

    try:
        contig2pcs[conitg].append((idx, pc, evalue))
    except:
        contig2pcs[conitg] = [(idx, pc, evalue)]


for contig in contig2pcs:
    contig2pcs[contig] = sorted(contig2pcs[contig], key=lambda tup: int(tup[0]))
pkl.dump(contig2pcs, open(f'{eval_fn}/contig2pcs.dict', 'wb'))

# Contigs2sentence
contig2id = {contig: idx for idx, contig in enumerate(contig2pcs.keys())}
id2contig = {idx: contig for idx, contig in enumerate(contig2pcs.keys())}

sentence = np.zeros((len(contig2id.keys()), 100))

sentence_weight = np.ones((len(contig2id.keys()), 100))

for row in range(sentence.shape[0]):
    contig = id2contig[row]
    pcs = contig2pcs[contig]
    for col in range(len(pcs)):
        try:
            _, sentence[row][col], sentence_weight[row][col] = pcs[col]
            sentence[row][col] += 1
        except:
            break


pkl.dump(sentence, open(f'{eval_fn}/sentence.feat', 'wb'))
pkl.dump(id2contig, open(f'{eval_fn}/id2contig.dict', 'wb'))

