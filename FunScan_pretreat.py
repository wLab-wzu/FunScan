#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import subprocess
import argparse



parser = argparse.ArgumentParser(description="""FunScan_pretreat""")
parser.add_argument('--contigs', help='FASTA file of contigs', default='FunScan_data/mmseq3.fasta')
parser.add_argument('--inputfolder', help='FASTA mkfile of contigs', type=str, default='FunScan_data/')
parser.add_argument('--proteins', help='FASTA file of predicted proteins (optional)')
parser.add_argument('--midfolder', help='folder to store the intermediate files', type=str, default='FunScan_pretreat/')
parser.add_argument('--threads', help='number of threads to use', type=int, default=64)
inputs = parser.parse_args()

int_fn = inputs.inputfolder
threads = inputs.threads
out_fn = inputs.midfolder
if not os.path.isdir(out_fn):
    os.makedirs(out_fn)


diamond_path = '/home/zh22tlsun/anaconda3/envs/diamond/bin/diamond'
diamond_db = f'{int_fn}/mmseq3.dmnd'

try:
    diamond_cmd = f'{diamond_path} blastp --threads {threads} --sensitive  -d {diamond_db} -q {inputs.contigs} -o {out_fn}/mmseq3.tab -k 0 -e 1e-3'
    print("Running Diamond...")
    _ = subprocess.check_call(diamond_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    diamond_out_fp = f"{out_fn}/mmseq3.tab"
    database_abc_fp = f"{out_fn}/mmseq3.abc"
    _ = subprocess.check_call("awk '{{print $1,$2,$11}}' {0} > {1}".format(diamond_out_fp, database_abc_fp), shell=True)
except:
    print("diamond blastp failed")
    exit(1)


def generate_gene2genome():
    # 添加阈值后的.abc文件 sep='\t' 不添加阈值的.abc文件 修改为sep=' '
    blastp = pd.read_csv(f'{out_fn}/mmseq3.abc', sep=' ', names=["contig", "ref", "e-value"])
    contig_id = [item.rsplit("_", 1)[0] for item in blastp["contig"]]
    description = ["hypothetical protein" for _ in blastp["contig"]]
    gene2genome = pd.DataFrame({"protein_id": blastp["ref"], "contig_id": contig_id, "keywords": description})



generate_gene2genome()
database_abc_fp = f'{out_fn}/mmseq3.abc'
gene2genome_fp = f"{out_fn}/contig_gene_to_genome_fungi_c.csv"
gene2genome_df = pd.read_csv(gene2genome_fp, sep=',', header=0)
pc_inflation = 2.0


def make_protein_clusters_mcl(abc_fp, pc_inflation):
    print("Running MCL...")
    inflation = pc_inflation
    abc_fn = "merged"
    mci_fn = '{}.mci'.format(abc_fn)
    mci_fp = os.path.join('', mci_fn)
    mcxload_fn = '{}_mcxload.tab'.format(abc_fn)
    mcxload_fp = os.path.join('', mcxload_fn)
    subprocess.check_call("mcxload -abc {0} --stream-mirror --stream-neg-log10 -stream-tf 'ceil(200)' -o {1}"
                          " -write-tab {2}".format(abc_fp, mci_fp, mcxload_fp), shell=True)
    mcl_clstr_fn = "{0}_mcl{1}.clusters".format(abc_fn, int(inflation * 10))
    mcl_clstr_fp = os.path.join('', mcl_clstr_fn)
    subprocess.check_call("mcl {0} -I {1} -use-tab {2} -o {3}".format(
        mci_fp, inflation, mcxload_fp, mcl_clstr_fp), shell=True)
    return mcl_clstr_fp


def build_clusters(fp, gene2genome):
    """
        Build clusters given clusters file

        Args:
            fp (str): filepath of clusters file
            gene2genome (dataframe): A dataframe giving the protein and its genome.
            mode (str): clustering method
        Returns:
            tuple: dataframe of proteins, clusters, profiles and contigs
        """
    # Read MCL
    clusters_df, name, c = load_mcl_clusters(fp)
    print("Using MCL to generate PCs c.")
    # Assign each prot to its cluster
    gene2genome.set_index("protein_id", inplace=True)  # id, contig, keywords, cluster
    for prots, clust in zip(c, name):
        try:
            gene2genome.loc[prots, "cluster"] = clust
        except KeyError:
            prots_in = [p for p in prots if p in gene2genome.index]
            not_in = frozenset(prots) - frozenset(prots_in)
            print("{} protein(s) without contig: {}".format(len(not_in), not_in))
            gene2genome.loc[prots_in, "cluster"] = clust
    for clust, prots in gene2genome.groupby("cluster"):
        clusters_df.loc[clust, "annotated"] = prots.keywords.count()
        if prots.keywords.count():
            keys = ";".join(prots.keywords.dropna().values).split(";")
            key_count = {}
            for k in keys:
                k = k.strip()
                try:
                    key_count[k] += 1
                except KeyError:
                    key_count[k] = 1
            clusters_df.loc[clust, "keys"] = "; ".join(["{} ({})".format(x, y) for x, y in key_count.items()])
    gene2genome.reset_index(inplace=True)
    clusters_df.reset_index(inplace=True)
    profiles_df = gene2genome.loc[:, ["contig_id", "cluster"]].drop_duplicates()
    profiles_df.columns = ["contig_id", "pc_id"]
    contigs_df = pd.DataFrame(gene2genome.fillna(0).groupby("contig_id").count().protein_id)
    contigs_df.index.name = "contig_id"
    contigs_df.columns = ["proteins"]
    contigs_df.reset_index(inplace=True)

    return gene2genome, clusters_df, profiles_df, contigs_df


def load_mcl_clusters(fi):
    with open(fi) as f:
        c = [line.rstrip("\n").split("\t") for line in f]
    # 将簇小于1的去除 降低过拟合
    c = [x for x in c if len(x) > 10]

    nb_clusters = len(c)
    formatter = "PC_{{:>0{}}}".format(int(round(np.log10(nb_clusters)) + 1))
    name = [formatter.format(str(i)) for i in range(nb_clusters)]
    size = [len(i) for i in c]
    clusters_df = pd.DataFrame({"size": size, "pc_id": name}).set_index("pc_id")
    return clusters_df, name, c


pcs_fp = make_protein_clusters_mcl(database_abc_fp, pc_inflation)  # protein_network.gml
protein_df, clusters_df, profiles_df, contigs_df = build_clusters(pcs_fp, gene2genome_df)
dfs0 = [gene2genome_df, contigs_df, clusters_df]
names = ['proteins', 'contigs', 'pcs']

for name, df in zip(names, dfs0):
    fn = f'{out_fn}/train_funscan_{name}.csv'
    fp = os.path.join('', fn)
    index_id = name.strip('s') + '_id'
    df.set_index(index_id).to_csv(fp)