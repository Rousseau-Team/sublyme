# Predict lysins in a given dataset of phage proteins

import os
import time
import pickle
import joblib
import argparse
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from multiprocessing.dummy import Pool as ThreadPool

if __package__ is None or __package__ == '':
    from embeddings import *
    preloaded_models=False
else:
    from .embeddings import *
    from importlib import resources
    clf1_path = str(resources.files('sublyme.models').joinpath('lysin_miner.pkl'))
    clf2_path = str(resources.files('sublyme.models').joinpath('val_endo_clf.pkl'))
    preloaded_models=True

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='Path to input file containing protein sequences (.fa*) or protein embeddings (.csv) that you wish to annotate.')
    parser.add_argument('-o', '--output_folder', help='Path to the output folder. Default folder is ./outputs.', default="./outputs/")
    parser.add_argument('-m', '--models_folder', help='Path to folder containing pretrained models. Default is src/sublme/models.', default="./src/sublyme/models")
    parser.add_argument('-t', '--threads', type=int, help='Number of threads. Default 1.', default=1)
    parser.add_argument('--only_embeddings', help='Whether to only calculate embeddings (no functional prediction).', action='store_true')
    args = parser.parse_args()


    input_file = args.input_file
    models_folder = args.models_folder
    only_embeddings = args.only_embeddings
    output_folder = args.output_folder
    threads = args.threads

    return input_file, models_folder, only_embeddings, output_folder, threads


# Load dataset we want to make predictions for
def load_dataset(input_file, threads):

    print("Loading dataset...")

    X_tests = []

    if threads ==1:
        if input_file.endswith(".pkl"):
            test = pd.read_pickle(input_file)

        if input_file.endswith(".csv"):
            test = pd.read_csv(input_file, index_col=0)
            test.columns = test.columns.astype(int)

        X_tests.append(test.loc[:, 0:1023])

    else:
        if input_file.endswith(".pkl"):
            test = pd.read_pickle(input_file)
            X_tests.append(test.loc[:, 0:1023])

        if input_file.endswith(".csv"):
            f = np.memmap(input_file)
            file_lines = sum(np.sum(f[i:i+(1024*1024)] == ord('\n')) for i in range(0, len(f), 1024*1024))
            chunk_size = (file_lines // threads) + 1

            for chunk in pd.read_csv(input_file, index_col=0, chunksize=chunk_size):
                chunk.columns = chunk.columns.astype(int)
                chunk = chunk.loc[:, 0:1023]
                X_tests.append(chunk)

    print("Done loading dataset.")

    return X_tests


def calc_embeddings(input_file, output_folder):

    lookup_p = Path(input_file)
    output_d = Path(output_folder)

    start=time.time()
    processor = sequence_processor(lookup_p, output_d)

    end=time.time()

    print("Total time: {:.3f}[s] ({:.3f}[s]/protein)".format(
        end-start, (end-start)/len(processor.lookup_ids)))

    return None


def predict(data, models_folder):
    data = data.loc[:, 0:1023]

    #load classifiers
    if preloaded_models==False:
        clf1 = joblib.load(os.path.join(models_folder, "lysin_miner.pkl"))
        clf2 = joblib.load(os.path.join(models_folder, "val_endo_clf.pkl"))
    else:
        clf1 = joblib.load(clf1_path)
        clf2 = joblib.load(clf2_path)

    #make predictions clf1
    preds = pd.DataFrame(data=clf1.predict_proba(data)[:,1], columns=["lysin"], index=data.index)
    lysins = preds.loc[preds["lysin"] > 0.5, :].index
    non_lysins = preds.loc[preds["lysin"] <= 0.5, :].index

    preds["endolysin"] = None; preds["val"] = None; preds["pred"] = None
    if len(lysins) > 0: #select only prots predicted as lysins
        # make preds clf2 and merge with clf1 preds
        preds2 = pd.DataFrame(data=clf2.predict_proba(data.loc[lysins]), columns=clf2.classes_, index=lysins)
        preds = preds.combine_first(preds2)

        # format output
        assign = pd.DataFrame(columns=preds.columns, index=preds.index)
        for i in assign.columns: #assign final prediction per row
            assign[i] = np.where(preds[i] > 0.5, i, "")
        preds["pred"] = assign["lysin"] + "|" + assign["endolysin"] + "|" + assign["val"]
        preds["pred"] = preds["pred"].str.replace(" ","").str.strip("|")

    return preds.loc[:, ["pred", "lysin", "endolysin", "val"]]


# Save predictions
def save_preds(preds, output_folder):

    print("Saving predictions to file...")

    preds[0].to_csv(os.path.join(output_folder, f"sublyme_predictions.csv"))
    for pred in preds[1:]:
        pred.to_csv(os.path.join(output_folder, f"sublyme_predictions.csv"), mode="a", header=False)

    print("Done saving predictions to file.")

def launch_per_thread(X_test, models_folder):

    #Remove entries with duplicate names
    if X_test.index.duplicated().sum() > 0:
        print(X_test.index.duplicated().sum(), "sequences with duplicate names were removed. Make sure this is normal as you may have lost some sequences. Here is the list of problematic IDs:", X_test[X_test.index.duplicated()].index)
    X_test = X_test.loc[~X_test.index.duplicated()]

    #Make predictions
    preds = predict(X_test, models_folder)

    return preds


#Main function. Loads dataset and makes predictions.
def lysin_miner(input_file, models_folder="models", only_embeddings=False, output_folder="outputs", threads=1):

    #Create output folder
    if not os.path.exists(os.path.join(output_folder)):
        os.makedirs(os.path.join(output_folder))

    #Load dataset
    if input_file.endswith((".fa", ".faa", ".fasta")): #input are protein sequences
        calc_embeddings(input_file, output_folder) #compute embeddings and save to file
        if only_embeddings:
            return None #stop before making predictions
        fname = f"{os.path.split(input_file)[1].rsplit('.', 1)[0]}.csv"
        X_tests = load_dataset(os.path.join(output_folder, fname), threads)

    elif input_file.endswith((".pkl", ".csv")): #input are protein embeddings
        X_tests = load_dataset(input_file, threads)

    else:
        print("Input file provided does not have an accepted extension (.pkl, .csv, .fa, .faa, .fasta).")

    pool = ThreadPool(threads)
    results = pool.starmap(launch_per_thread, zip(X_tests, itertools.repeat(models_folder, len(X_tests)) ))

    save_preds(results, output_folder)


def main():
    #Load user args
    input_file, models_folder, only_embeddings, output_folder, threads = parse_args()
    lysin_miner(input_file, models_folder, only_embeddings, output_folder, threads)

if __name__ == '__main__':
    main()
    #Load user args
    #input_file, models_folder, only_embeddings, output_folder, threads = parse_args()
    #lysin_miner(input_file, models_folder, only_embeddings, output_folder, threads)
