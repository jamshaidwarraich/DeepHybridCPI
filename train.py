# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay, auc
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.lines as mlines

from metrics import accuracy, precision, auc_score, recall
from dataset import *
from model import MGraphCPI
from utils import *
from log.train_logger import TrainLogger

from utils import set_seed



def evaluate(model, criterion, dataloader, device, extract_features=False):
    model.eval()
    running_loss = AverageMeter()

    pred_list = []
    pred_cls_list = []
    label_list = []
    embedding_list = []

    for data in dataloader:
        data.y = data.y.long()
        data = data.to(device)

        with torch.no_grad():
            if extract_features:
                embedding = model(data, return_embedding=True)
                embedding_list.append(embedding.cpu().numpy())
                label_list.append(data.y.cpu().numpy())
                continue

            pred = model(data)
            loss = criterion(pred, data.y)
            pred_cls = torch.argmax(pred, dim=-1)

            pred_prob = F.softmax(pred, dim=-1)
            pred_prob, indices = torch.max(pred_prob, dim=-1)
            pred_prob[indices == 0] = 1. - pred_prob[indices == 0]

            pred_list.append(pred_prob.view(-1).cpu().numpy())
            pred_cls_list.append(pred_cls.view(-1).cpu().numpy())
            label_list.append(data.y.cpu().numpy())
            running_loss.update(loss.item(), data.y.size(0))

    if extract_features:
        embeddings = np.concatenate(embedding_list, axis=0)
        labels = np.concatenate(label_list, axis=0)
        return embeddings, labels

    pred = np.concatenate(pred_list, axis=0)
    pred_cls = np.concatenate(pred_cls_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    acc = accuracy(label, pred_cls)
    pre = precision(label, pred_cls)
    rec = recall(label, pred_cls)
    auc_val = auc_score(label, pred)

    loss = running_loss.get_average()
    running_loss.reset()

    return loss, acc, pre, rec, auc_val, pred, pred_cls, label



def run_single_training(params, seed):
    """Run one full training + evaluation with a specific seed."""
    set_seed(seed)

    logger = TrainLogger(params)
    logger.info(f"RUN WITH SEED = {seed}")
    logger.info(__file__)

    DATASET = params["dataset"]
    save_model_flag = params["save_model"]
    data_root = params["data_root"]
    fpath = os.path.join(data_root, DATASET)

    # Load dataset
    train_set = GNNDataset(fpath, types='train')
    val_set = GNNDataset(fpath, types='val')
    test_set = GNNDataset(fpath, types='test')

    train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_set, batch_size=params['batch_size'], shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_set, batch_size=params['batch_size'], shuffle=False, num_workers=0)


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = MGraphCPI(
        embedding_size=128,
        num_filters=32,
        out_dim=2
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_model_path = os.path.join("save", f"best_model_seed_{seed}.pt")

    # Tracking
    for epoch in range(params['epochs']):
        model.train()
        running_loss = AverageMeter()
        all_train_preds = []
        all_train_labels = []

        for data in train_loader:
            data.y = data.y.long()
            data = data.to(device)

            pred = model(data)
            loss = criterion(pred, data.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.update(loss.item(), data.y.size(0))

            pred_cls = torch.argmax(pred, dim=-1)
            all_train_preds.extend(pred_cls.cpu().numpy())
            all_train_labels.extend(data.y.cpu().numpy())

        train_acc = accuracy(np.array(all_train_labels), np.array(all_train_preds))

        # Validation
        val_loss, val_acc, _, _, _, _, _, _ = evaluate(model, criterion, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"[Seed {seed}] New best model saved with val_acc={best_val_acc:.4f}")

    # --- Test evaluation ---
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, test_pre, test_rec, test_auc, _, _, _ = evaluate(model, criterion, test_loader, device)

    return test_acc, test_pre, test_rec, test_auc



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    params = dict(
        data_root="data",
        save_dir="save",
        dataset=args.dataset,
        save_model=args.save_model,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs
    )

    # ---- RUN 3 TIMES WITH DIFFERENT SEEDS ----
    seeds = [42, 123, 999]

    acc_list = []
    pre_list = []
    rec_list = []
    auc_list = []

    for seed in seeds:
        print(f"\n========== RUNNING SEED {seed} ==========\n")
        acc, pre, rec, auc_val = run_single_training(params, seed)

        acc_list.append(acc)
        pre_list.append(pre)
        rec_list.append(rec)
        auc_list.append(auc_val)

    # ---- PRINT FINAL RESULTS (MEAN ± STD) ----
    print("\n\n======== FINAL 3-RUN RESULTS ========")
    print(f"Accuracy:  {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    print(f"Precision: {np.mean(pre_list):.4f} ± {np.std(pre_list):.4f}")
    print(f"Recall:    {np.mean(rec_list):.4f} ± {np.std(rec_list):.4f}")
    print(f"AUC:       {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")


if __name__ == "__main__":
    main()
