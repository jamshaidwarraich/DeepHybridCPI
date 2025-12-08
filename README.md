# DeepHybridCPI: A Hybrid Deep Learning Framework for Compound-Protein Interaction Prediction

## Dataset

All data used in this paper are publicly available and can be accessed here: https://github.com/masashitsubaki/CPI_prediction  

## Requirements  

pip install torch_geometric
pip install rdkit

## Train/test DeepHybridCPI:
  
- First, run preprocessing.py using  
  `python preprocessing.py`  

- Second, run train.py using 
  `python train.py --dataset human --save_model` for Human dataset and `python train.py --dataset celegans --save_model` for C.elegans dataset

