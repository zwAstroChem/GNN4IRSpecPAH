import pickle
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import AttentiveFP
from rdkit import Chem
import numpy as np
import os
import statistics

# -----------------------
# Automatically select device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -----------------------
# Standard EMD Loss (per-sample normalization)
# -----------------------
def emd_loss(pred, target):
    pred_sum = pred.sum(dim=1, keepdim=True) + 1e-8
    target_sum = target.sum(dim=1, keepdim=True) + 1e-8

    pred = pred / pred_sum
    target = target / target_sum

    pred_cdf = torch.cumsum(pred, dim=1)
    target_cdf = torch.cumsum(target, dim=1)

    sample_emd = torch.sum(torch.abs(pred_cdf - target_cdf), dim=1)
    return torch.mean(sample_emd)


# -----------------------
# Spectrum normalization
# -----------------------
def normalize_spectrum(s):
    s = np.array(s, dtype=np.float32)
    s_max = s.max()
    if s_max > 0:
        s /= s_max
    return s

'''
# Sum normalization
def normalize_spectrum(s):
    s = np.array(s, dtype=np.float32)   # convert to float32 array
    s_sum = s.sum()                     # total spectral intensity
    if s_sum > 0:                       # avoid division by zero
        s /= s_sum                      # divide each point by total sum
    return s
'''
'''
# Global normalization
def normalize_spectrum(s, global_max):
    s = np.array(s, dtype=np.float32)
    if global_max > 0:
        s = s / global_max
    return s
'''
# -----------------------
# SMILES → PyG Data (with smiles stored)
# -----------------------
def mol_to_graph_data(smiles, spectrum):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_feats = []
    for atom in mol.GetAtoms():
        at_type = atom.GetAtomicNum()
        feat = [0] * 11
        if at_type == 1:
            feat[0] = 1
        elif at_type == 6:
            feat[1] = 1
        elif at_type == 8:
            feat[2] = 1
        elif at_type == 7:
            feat[3] = 1
        elif at_type == 16:
            feat[4] = 1
        elif at_type == 9:
            feat[5] = 1
        elif at_type == 17:
            feat[6] = 1
        elif at_type == 35:
            feat[7] = 1
        elif at_type == 53:
            feat[8] = 1
        elif at_type == 15:
            feat[9] = 1
        else:
            feat[10] = 1
        feat.append(atom.GetDegree())
        atom_feats.append(feat)
    x = torch.tensor(atom_feats, dtype=torch.float)

    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])

        bt = bond.GetBondType()
        bt_feat = [0, 0, 0, 0]
        if bt == Chem.rdchem.BondType.SINGLE:
            bt_feat[0] = 1
        elif bt == Chem.rdchem.BondType.DOUBLE:
            bt_feat[1] = 1
        elif bt == Chem.rdchem.BondType.TRIPLE:
            bt_feat[2] = 1
        elif bt == Chem.rdchem.BondType.AROMATIC:
            bt_feat[3] = 1
        edge_attr.append(bt_feat)
        edge_attr.append(bt_feat)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    y = torch.tensor(normalize_spectrum(spectrum), dtype=torch.float).unsqueeze(0)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smiles)
    return data


# -----------------------
# Data loading
# -----------------------
with open("3.2_CH_Cleaner PAHs Dataset.pickle", "rb") as f:
    d = pickle.load(f)

smiles_list = d["smiles"]
spectra_list = d["sequences"]

# global_max = max(spec.max() for spec in spectra_list if len(spec) > 0)

dataset = []
for s, spec in zip(smiles_list, spectra_list):
    data = mol_to_graph_data(s, spec)
    if data is not None:
        dataset.append(data)

np.random.seed(13)
np.random.shuffle(dataset)

# -----------------------
# Five-fold split
# -----------------------
n = len(dataset)
fold_size = n // 5
folds = [dataset[i * fold_size:(i + 1) * fold_size] for i in range(4)]
folds.append(dataset[4 * fold_size:])

# -----------------------
# Automatically detect dimensions
# -----------------------
sample_data = dataset[0]
node_feat_dim = sample_data.x.shape[1]
edge_feat_dim = sample_data.edge_attr.shape[1]
spectrum_len = sample_data.y.shape[1]

print(f"Node features: {node_feat_dim}, Edge features: {edge_feat_dim}, Spectrum length: {spectrum_len}")


# -----------------------
# AFP model definition
# -----------------------
class AFP_Spectra_Model(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_dim, edge_dim, num_layers=2, num_timesteps=2, dropout=0.1):
        super(AFP_Spectra_Model, self).__init__()
        self.afp = AttentiveFP(
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            out_channels=out_dim,
            edge_dim=edge_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout
        )

    def forward(self, data):
        return self.afp(data.x, data.edge_index, data.edge_attr, batch=data.batch)


# -----------------------
# Training & evaluation functions
# -----------------------
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        preds = model(batch)
        loss = emd_loss(preds, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def evaluate(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch)
            loss = emd_loss(preds, batch.y)
            total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


# -----------------------
# Main training loop: five-fold cross-validation + early stopping
# -----------------------
MAX_EPOCHS = 500
PATIENCE = 300
output_dir = "./Fold_Predictions"
os.makedirs(output_dir, exist_ok=True)

for fold_idx in range(5):
    print(f"\n========== Fold {fold_idx + 1} ==========")

    test_dataset = folds[fold_idx]
    train_dataset = [item for i, f in enumerate(folds) if i != fold_idx for item in f]

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = AFP_Spectra_Model(
        in_channels=node_feat_dim,
        hidden_dim=128,
        out_dim=spectrum_len,
        edge_dim=edge_feat_dim,
        num_layers=3,
        num_timesteps=2,
        dropout=0.1
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_emd = float('inf')
    wait = 0
    best_model_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss = evaluate(model, test_loader)
        print(f"Epoch {epoch:03d} | Train EMD: {train_loss:.4f} | Val EMD: {val_loss:.4f}")

        if val_loss < best_val_emd:
            best_val_emd = val_loss
            best_model_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_model_state)

    # -----------------------
    # Mean ± standard deviation of EMD on test set for each fold
    # -----------------------
    model.eval()
    emd_list = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds = model(batch)
            for i in range(batch.num_graphs):
                emd_list.append(emd_loss(preds[i:i + 1], batch.y[i:i + 1]).item())

    avg_emd = statistics.mean(emd_list)
    std_emd = statistics.stdev(emd_list) if len(emd_list) > 1 else 0.0
    print(f"Low_AFP_1_Fold {fold_idx + 1} Test EMD: {avg_emd:.5f} ± {std_emd:.5f}")

    # -----------------------
    # Save predictions (with smiles and ground truth)
    # -----------------------
    results = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds = model(batch).cpu().numpy()
            trues = batch.y.cpu().numpy()
            smiles_list = batch.smiles  # directly retrieve stored SMILES
            for p, t, s in zip(preds, trues, smiles_list):
                results.append({
                    "smiles": s,
                    "pred": p,
                    "true": t
                })

    fold_path = os.path.join(output_dir, f"ALL_High_AFP_1_Fold_{fold_idx + 1}_results.pickle")
    with open(fold_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"ALL_High_AFP_1_Fold {fold_idx + 1} results saved to {fold_path}")
