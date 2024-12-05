# Predict lysins in a given dataset of phage proteins

import os
import time
import pickle
import joblib
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC

if __package__ is None or __package__ == '':
    from embeddings import *
else:
    from .embeddings import *


# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='Path to input file containing protein sequences (.fa*) or protein embeddings (.pkl/.csv) that you wish to annotate (.pkl/.csv).')
    parser.add_argument('--only_embeddings', help='Whether to only calculate embeddings (no functional prediction).', action='store_true')
    parser.add_argument('-f', '--models_folder', help='Path to folder containing pretrained models. Default folder is ./models/', default="./models/")
    parser.add_argument('-o', '--output_folder', help='Path to the output folder. Default folder is ./outputs/', default="./outputs/")
    args = parser.parse_args()


    input_file = args.input_file
    models_folder = args.models_folder
    only_embeddings = args.only_embeddings
    output_folder = args.output_folder

    return input_file, models_folder, only_embeddings, output_folder


# Load dataset we want to make predictions for
def load_dataset(input_file):

    print("Loading dataset...")

    if input_file.endswith(".pkl"):
        test = pd.read_pickle(input_file)

    if input_file.endswith(".csv"):
        test = pd.read_csv(input_file, index_col=0)
        test.columns = test.columns.astype(int)

    X_test = test.loc[:, 0:1023]

    print("Done loading dataset.")

    return X_test


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
    clf1 = joblib.load(os.path.join(models_folder, "lysin_miner.pkl"))
    clf2 = joblib.load(os.path.join(models_folder, "val_endo_clf.pkl"))

    preds = pd.DataFrame(data=clf1.predict_proba(data)[:,1], columns=["lysin"], index=data.index)
    lysins = preds.loc[preds["lysin"] > 0.5, :].index
    non_lysins = preds.loc[preds["lysin"] <= 0.5, :].index

    if len(lysins) > 0:
        preds2 = pd.DataFrame(data=clf2.predict_proba(data.loc[lysins]), columns=clf2.classes_, index=lysins)
        return pd.merge(preds, preds2, left_index=True, right_index=True, how="outer").fillna(0)

    return preds


# Save predictions
def save_preds(preds, output_folder): #name

    print("Saving predictions to file...")

    preds.to_csv(os.path.join(output_folder, f"predictions_sublyme.csv")) #name,

    print("Done saving predictions to file.")


#Main function. Loads dataset and makes predictions.
def lysin_miner(input_file, models_folder="models", only_embeddings=False, output_folder="outputs"):

    #Create output folder
    if not os.path.exists(os.path.join(output_folder)):
        os.makedirs(os.path.join(output_folder))

    #Load dataset
    if input_file.endswith((".fa", ".faa", ".fasta")): #input are protein sequences
        calc_embeddings(input_file, output_folder) #compute embeddings and save to file
        if only_embeddings:
            return None #stop before making predictions
        fname = f"{os.path.split(input_file)[1].rsplit('.', 1)[0]}.csv"
        X_test = load_dataset(os.path.join(output_folder, fname))

    elif input_file.endswith((".pkl", ".csv")): #input are protein embeddings
        X_test = load_dataset(input_file)

    else:
        print("Input file provided does not have an accepted extension (.pkl, .csv, .fa, .faa, .fasta).")

    #Remove entries with duplicate names
    if X_test.index.duplicated().sum() > 0:
        print(X_test.index.duplicated().sum(), "sequences with duplicate names were removed. Make sure this is normal as you may have lost some sequences. Here is the list of problematic IDs:", X_test[X_test.index.duplicated()].index)
    X_test = X_test.loc[~X_test.index.duplicated()]

    #Make predictions
    preds = predict(X_test, models_folder)

    save_preds(preds, output_folder)


if __name__ == '__main__':
    #Load user args
    input_file, models_folder, only_embeddings, output_folder = parse_args()
    lysin_miner(input_file, models_folder, only_embeddings, output_folder)
