#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:38:40 2021 (modified Wed Apr 3 09:55:00 2024)

@author: mheinzinger adapted by aboulay
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch import nn

import random
from sklearn.metrics import accuracy_score, balanced_accuracy_score,  f1_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class Embedder():
    def __init__(self):
        self.embedder, self.tokenizer = self.get_prott5()

    def get_prott5(self):
        start=time.time()
        # Load your checkpoint here
        # Currently, only the encoder-part of ProtT5 is loaded in half-precision
        from transformers import T5EncoderModel, T5Tokenizer
        print("Start loading ProtT5...")
        transformer_name = "Rostlab/prot_t5_xl_half_uniref50-enc"
        model = T5EncoderModel.from_pretrained(transformer_name, torch_dtype=torch.float32)
        model = model.to(device)
        model = model.eval()
        tokenizer = T5Tokenizer.from_pretrained(transformer_name, do_lower_case=False)
        print("Finished loading {} in {:.1f}[s]".format(transformer_name,time.time()-start))
        return model, tokenizer

    def get_embeddings_batch(self, id2seq, max_residues=4000, max_seq_len=1000, max_batch=100):
        print("Start generating embeddings for {} proteins.".format(len(id2seq)) +
              "This process might take a few minutes." +
              "Using batch-processing! If you run OOM/RuntimeError, you should use single-sequence embedding by setting max_batch=1.")
        start = time.time()
        ids = list()
        embeddings = list()
        batch = list()

        id2seq = sorted( id2seq.items(), key=lambda kv: len( id2seq[kv[0]] ), reverse=True )
        for seq_idx, (protein_id, original_seq) in enumerate(id2seq):
            seq = original_seq.replace('U','X').replace('Z','X').replace('O','X')
            seq_len = len(seq)
            seq = ' '.join(list(seq))
            batch.append((protein_id,seq,seq_len))
            print("Dealing with", seq_idx,"over",len(id2seq),"prots")
            n_res_batch = sum([ s_len for  _, _, s_len in batch ]) + seq_len 
            if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(id2seq)-1 or seq_len>max_seq_len:
                protein_ids, seqs, seq_lens = zip(*batch)
                batch = list()

                token_encoding = self.tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
                input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
                attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)
                print("Going to compute embeddings...")
                try:
                    with torch.no_grad():
                        # get embeddings extracted from last hidden state 
                        batch_emb = self.embedder(input_ids, attention_mask=attention_mask).last_hidden_state # [B, L, 1024]
                except RuntimeError as e :
                    print(e)
                    print("RuntimeError during embedding for {} (L={})".format(protein_id, seq_len))
                    continue

                for batch_idx, identifier in enumerate(protein_ids):
                    s_len = seq_lens[batch_idx]
                    emb = batch_emb[batch_idx,:s_len].mean(dim=0,keepdims=True)
                    ids.append(protein_ids[batch_idx])
                    embeddings.append(emb.detach())

        print("Creating per-protein embeddings took: {:.1f}[s]".format(time.time()-start))
        embeddings = torch.vstack(embeddings)
        return ids, embeddings


# process protein sequences
class sequence_processor():
    def __init__(self, lookup_p, output_d):

        self.output_d = output_d
        Path.mkdir(output_d, exist_ok=True)

        self.Embedder = None 

        self.lookup_ids, self.lookup_embs = self.read_inputs(lookup_p)


    def read_inputs(self, input_p):
        # define path for storing embeddings

        if not input_p.is_file():
            print("No input fasta could be found for: {}".format(input_p))
            print("Files are expected to end with .fasta.")
            raise FileNotFoundError


        if input_p.name.endswith(".fasta") or input_p.name.endswith(".faa"): # compute new embeddings if only FASTA available
            if self.Embedder is None: # avoid re-loading the pLM
                self.Embedder = Embedder()
            id2seq = self.read_fasta(input_p)

            ids, embeddings = self.Embedder.get_embeddings_batch(id2seq)

            emb_p  = self.output_d / input_p.name.replace(".faa", ".csv")

            embeddings_=embeddings.detach().cpu().numpy().squeeze()
            with open(emb_p,'wb') as f:
                pd.DataFrame(data=embeddings_, index=ids, columns=range(0,1024)).to_csv(f)

            return ids, embeddings
        else:
            print("The file you passed did not end with .fasta. " +
                  "Only that file format is currently supported.")
            raise NotImplementedError

    def read_fasta(self, fasta_path):
        '''
            Store sequences in fasta file as dictionary with keys being fasta headers and values being sequences.
            Also, replace gap characters and insertions within the sequence as those can't be handled by ProtT5 
                when generating embeddings from sequences'.
            Also, replace special characters in the FASTA headers as those are interpreted as special tokens 
                when loading pre-computed embeddings from H5. 
        '''
        sequences = dict()
        with open(fasta_path, 'r') as fasta_f:
            for line in fasta_f:
                line=line.strip()
                if not len(line):
                    continue
                # get uniprot ID from header and create new entry
                if line.startswith('>'):
                    #if '|' in line and (line.startswith(">tr") or line.startswith(">sp")):
                    #    seq_id = line.split("|")[1]
                    #else:
                    #    seq_id = line.replace(">","")
                    seq_id = line[1:].split(" ")[0]
                    # replace tokens that are mis-interpreted when loading h5
                    #seq_id = seq_id.replace("/", "_").replace(".", "_")
                    sequences[seq_id] = ''
                else:
                    # repl. all whie-space chars and join seqs spanning multiple lines
                    # drop gaps and cast to upper-case
                    sequences[seq_id] += ''.join(
                        line.split()).upper().replace("-", "")
        return sequences

def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""

    # Instantiate the parser
    parser = argparse.ArgumentParser(description=(
            """
                The input (lookup) should be passed as raw protein sequence files (*.fasta).
            """
    ))

    # Required positional argument
    parser.add_argument('-l', '--lookup', required=True, type=str,
                        help='A path to your lookup file (*.fasta).')


    # Optional positional argument
    parser.add_argument('-o', '--output', required=True, type=str,
                        help='A path to folder storing EAT results.')


    return parser


def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    lookup_p = Path(args.lookup)
    output_d = Path(args.output)

    start=time.time()
    processor = sequence_processor(lookup_p, output_d)

    end=time.time()

    print("Total time: {:.3f}[s] ({:.3f}[s]/protein)".format(
        end-start, (end-start)/len(processor.lookup_ids)))

    return None


if __name__ == '__main__':
    main()
