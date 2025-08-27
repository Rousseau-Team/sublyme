<h1 align="center">SUBLYME</h1>
<div align="center"> <strong>S</strong>oftware for <strong>U</strong>ncovering <strong>B</strong>acteriophage <strong>LY</strong>sins in <strong>ME</strong>tagenomic datasets</div>
<br>

<!-- TABLE OF CONTENTS -->
<details open>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About the Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage-details">Usage details</a></li>
    <li><a href="#output-format">Output format</a></li>
  </ol>
</details>

## About the Project

SUBLYME is a tool to identify bacteriophage lysins. It utilizes the highly informative ProtT5
protein embeddings to make predictions and was trained using proteins in the [PHALP](https://phalp.ugent.be/) database.


## Getting started
SUBLYME has been packaged in [PyPI](https://pypi.org/project/sublyme/) for ease of use. The source code can be downloaded from [GitHub](https://github.com/Rousseau-Team/sublyme).


### Prerequisites
A GPU is recommended to compute embeddings for large datasets.

The full list of dependencies can be found in [requirements.txt](https://github.com/Rousseau-Team/sublyme/blob/main/requirements.txt).

Dependencies are taken care of by pip.
```
python/3.11.5
joblib==1.2.0
numpy==1.26.4
pandas==2.2.1
torch==2.3.0
scipy==1.13.1
scikit-learn==1.3.0
transformers==4.43.1
sentencepiece==0.2.0
```


### Installation

First create a virtual environment in python 3.11.5. For example:
```
conda create -n sublyme_env python=3.11.5
conda activate sublyme_env
```


**From pypi**:
```
pip install sublyme
```

Usage
```
sublyme test/input.faa -t 4
```


**From source**:
```
git clone https://github.com/Rousseau-Team/sublyme.git
cd sublyme
pip install -e requirements.txt
```

ex. `python3 src/sublyme/sublyme.py test/input.faa -t 4 --models_folder src/sublyme/models`


### Usage details
A fasta file of protein sequences or a csv file of protein embeddings can be used as input.

Specifying the option --only_embeddings will only compute embeddings. This step is much faster with a GPU.
The embeddings file can then be reinputted using the same command (without --only_embeddings) and specifying the new file as input file.

Options:
- **input_file**:           Path to input file containing protein sequences (.fa*) or protein embeddings (.csv) that you wish to annotate.
- **--threads** (-t):       Number of threads (default 1).
- **--output_folder** (-o): Path to the output folder. Default folder is ./outputs/.
- **--models_folder** (-m): Path to folder containing pretrained models (lysin_miner.pkl, val_endo_clf.pkl). Default is src/sublyme/models.
- **--only_embeddings**:    Whether to only calculate embeddings (no lysin prediction).

### Output format
The output consists of a csv file with a column for the final prediction and one column each for probabilities associated to lysins, endolysins and VALs. 

Ex.
|            pred           |lysin|endolysin|VAL |
|---------------------------|-----|---------|----|
|      lysin\|endolysin     |0.98 |0.95     |0.05|
|             Na            |0.01 |Na       |Na  |

Note that the endolysin/VAL classifier is one multiclass classifier, implying that their probabilities will always add up to one and that the classifier will always assign one of these to be true.

Also, the endolysin/VAL classifier is only applied to proteins first predicted as being lysins (lysin proba >0.5).

