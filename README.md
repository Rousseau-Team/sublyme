# SUBLYME: Software for Uncovering Bacteriophage LYsins in MEtagenomic datasets using protein embeddings


## Installation & usage

First create a virtual environment, then: 

**From pypi**:
```
pip install numpy, pandas, transformers, sublyme

python
from sublyme import sublyme
```

ex. `sublyme("seqs.faa", "./models/")`


**From source**:
```
git clone https://github.com/Rousseau-Team/sublyme.git

pip install numpy pandas transformers
```

ex. `python sublyme/src/sublyme/sublyme.py seqs.faa ./models/ --output_file ./outputs/`

## Advanced usage
**Positional arguments**:
- **input_file** - Path to input file containing protein sequences (.fa*) or protein embeddings (.pkl/.csv) that you wish to annotate (.pkl/.csv).
- **models_folder** -  Path to folder containing pretrained models (lysin_miner.pkl, endolysin_miner.pkl and val_miner.pkl).

**Optional arguments**:
- **output_file** - Path to the output folder. Default folder is ./outputs/.
- **only_embeddings** - Whether to only calculate embeddings (no lysin prediction).
