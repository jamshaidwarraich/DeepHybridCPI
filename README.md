# DeepHybridCPI: A Hybrid Deep Learning Framework for Compound-Protein Interaction Prediction

## ⚙️ Installation & Setup

###  💻 Google Colab (Recommended)

```bash
!pip install torch_geometric
!pip install rdkit
```

### Train/test DeepHybridCPI:
  
- First, run preprocessing.py using
```bash
  `python preprocessing.py`  
```

- Second, run train.py using:

 🧍 For Human dataset:
  ```bash
  python train.py --dataset human --save_model
  ```

 🐛 For C. elegans dataset:
  ```bash
  python train.py --dataset celegans --save_model
  ```

## 📊 Datasets

🌐 Public dataset used in this work: https://github.com/masashitsubaki/CPI_prediction 

