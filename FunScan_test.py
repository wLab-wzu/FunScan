
import subprocess
import os
import shutil
import torch
from torch import nn
from torch import optim
import numpy as np
import pandas as pd
import pickle as pkl
import argparse
from model import Transformer
from shutil import which
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
parser = argparse.ArgumentParser(description="""FunScan_test.""")
parser.add_argument('--inputdir', help='FASTA file of contigs', default='FunScan_data/')
parser.add_argument('--threads', help='number of threads to use', type=int, default=64)
parser.add_argument('--proteins', help='FASTA file of predicted proteins (optional)')
parser.add_argument('--proteinsfolder', help='protein vocabulary', type=str, default='FunScan_pretreat/')

parser.add_argument('--trainfolder', help='results file of train', type=str, default='FunScan_train_sentence/')
parser.add_argument('--testdir', help='test result file', type=str, default='FunScan_test_result/')
parser.add_argument('--testfolder', help='test folder to store the intermediate files', type=str, default='FunScan_test_mid/')
inputs = parser.parse_args()
input_fn = inputs.inputdir
protein_fn = inputs.proteinsfolder

train_fn = inputs.trainfolder
testout_fn = inputs.testdir
test_fn = inputs.testfolder
if not os.path.isdir(test_fn):
    os.makedirs(test_fn)
if not os.path.isdir(testout_fn):
    os.makedirs(testout_fn)
threads = inputs.threads

if inputs.proteins is None:
    prodigal = "prodigal"
    if which("pprodigal") is not None:
        print("Using parallelized prodigal...")
        prodigal = f'pprodigal -T {threads}'
    prodigal_cmd = f'{prodigal} -n -g 1 -p meta -i {input_fn}/test.rN.fa -a {test_fn}/test.faa -f gff -d {test_fn}/test.nuc.fna -T {threads}'
    _ = subprocess.check_call(prodigal_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
else:
    shutil.copyfile(inputs.proteins, f'{test_fn}/test.nuc.fna')

try:
    transeq_cmd = f'transeq -sequence {test_fn}/test.nuc.fna -outseq {test_fn}/test.nuc.transeq -frame F -clean Y'
    _ = subprocess.check_call(transeq_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except Exception as e:
    print(f"An unexpected error occurred: {e}")

try:
    perl_cmd = f"perl -ne 'chomp;if(/>(\\S+)_\\d(\\s+\\#.+)/){{ $i++;if($i eq 1){{ print \">$1 $2\\n\"; }}if($i eq 4){{ print \"$seq\\n\"; $seq=\"\"; $i=0; }} }}else{{ $seq.=$_; }}END{{ print \"$seq\\n\"; }}' {test_fn}/test.nuc.transeq > {test_fn}/test.nuc.transeq.s"
    _ = subprocess.check_call(perl_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except Exception as e:
    print(f"An unexpected error occurred: {e}")


diamond_path = '/home/zh22tlsun/anaconda3/envs/diamond/bin/diamond'
diamond_db = f'{input_fn}/mmseq3.dmnd'
try:
    diamond_cmd = f'{diamond_path} blastp --threads {threads} --sensitive  -d {diamond_db} -q {test_fn}/test.nuc.transeq.s -o {test_fn}/test.rN.nuc.tab -k 1 -e 1e-3'
    print("Running Diamond...")
    _ = subprocess.check_call(diamond_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    diamond_out_fp = f"{test_fn}/test.rN.nuc.tab"
    database_abc_fp = f"{test_fn}/test.rN.nuc.abc"
    _ = subprocess.check_call("awk '{{print $1,$2,$11}}' {0} > {1}".format(diamond_out_fp, database_abc_fp), shell=True)

except:
    print("diamond blastp failed")
    exit(1)

proteins_df = pd.read_csv(f'{protein_fn}/train_funscan_proteins')
proteins_df.dropna(axis=0, how='any', inplace=True)
pc2wordsid = {pc: idx for idx, pc in enumerate(sorted(set(proteins_df['cluster'].values)))}  # //映射簇名加簇名id
protein2pc = {protein: pc for protein, pc in zip(proteins_df['protein_id'].values, proteins_df['cluster'].values)}
blast_df = pd.read_csv(f"{test_fn}/test.rN.nuc.abc", sep=' ', names=['query', 'ref', 'evalue'])

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
pkl.dump(sentence, open(f'{test_fn}/sentence.feat', 'wb'))
pkl.dump(id2contig, open(f'{test_fn}/id2contig.dict', 'wb'))  # position

#################test#####################

pcs2idx = pkl.load(open(f'{train_fn}/pc2wordsid.dict', 'rb'))
num_pcs = len(set(pcs2idx.keys()))

device = torch.device("cuda")
if device.type == 'cpu':
    print("running with cpu")
    torch.set_num_threads(inputs.threads)
src_pad_idx = 0
src_vocab_size = num_pcs + 1

def reset_model():
    model = Transformer(
                src_vocab_size,
                src_pad_idx,
                device=device,
                max_length=100,
                dropout=0.5
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.BCEWithLogitsLoss()
    loss_func = loss_func.to(device)
    return model, optimizer, loss_func


def return_tensor(var, device):
    return torch.from_numpy(var).to(device)


model, optimizer, loss_func = reset_model()

try:
    pretrained_dict = torch.load(f'FunScan_transformer.pth', map_location=device)
    model.load_state_dict(pretrained_dict)
except:
    print('cannot find pre-trained model')
    exit()
model = model.to(device)



sentence   = pkl.load(open(f'{test_fn}/sentence.feat', 'rb'))
id2contig  = pkl.load(open(f'{test_fn}/id2contig.dict', 'rb'))

all_pred = []
all_score = []


with torch.no_grad():
    _ = model.eval()
    for idx in range(0, len(sentence), 500):
        try:
            batch_x = sentence[idx: idx+500]
        except:
            batch_x = sentence[idx:]
        batch_x = return_tensor(batch_x, device).long().to(device)
        logit = model(batch_x)
        logit = torch.sigmoid(logit.squeeze(1))
        pred = ['fungi' if item > 0.5 else 'other' for item in logit]
        all_pred += pred
        all_score += [float('{:.3f}'.format(i)) for i in logit]

csv_filename = f"{testout_fn}/test_result.csv"
pred_csv = pd.DataFrame({"Contig" : id2contig.values(), "Pred" : all_pred, "Score" : all_score})
pred_csv.to_csv(csv_filename, index = False)
