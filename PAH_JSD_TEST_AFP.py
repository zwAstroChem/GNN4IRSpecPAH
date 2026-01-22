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
# EMD Loss (for evaluation)
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


# -----------------------
# SMILES → PyG Data (with smiles attribute)
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
# AFP model definition (identical to training)
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
# Load new dataset (replace with your new dataset path)
# -----------------------
new_dataset_path = "4.0_CH_Cleaner 50_100_PAHs Dataset.pickle"  # Path to the new dataset
try:
    with open(new_dataset_path, "rb") as f:
        new_d = pickle.load(f)
    new_smiles_list = new_d["smiles"]
    new_spectra_list = new_d["sequences"]

    # Convert to PyG Data format
    new_dataset = []
    for s, spec in zip(new_smiles_list, new_spectra_list):
        data = mol_to_graph_data(s, spec)
        if data is not None:
            new_dataset.append(data)

    if not new_dataset:
        print("The new dataset is empty or conversion failed. Please check the data.")
        exit()

    # Automatically detect dimensions (should be consistent with training, or manually specified)
    sample_data = new_dataset[0]
    node_feat_dim = sample_data.x.shape[1]
    edge_feat_dim = sample_data.edge_attr.shape[1]
    spectrum_len = sample_data.y.shape[1]
    print(f"New dataset dimensions: node features {node_feat_dim}, edge features {edge_feat_dim}, spectrum length {spectrum_len}")

except FileNotFoundError:
    print(f"Error: new dataset not found at {new_dataset_path}")
    exit()


# -----------------------
# Load pretrained models and perform prediction
# -----------------------
model_weights_dir = "./Fold_Predictions/Best_Model"  # Directory containing model weights
all_fold_emd = []

for fold_idx in range(1, 6):
    print(f"\n===== Loading Fold {fold_idx} model =====")
    model_path = os.path.join(model_weights_dir, f"Low_JSD_1_Fold_{fold_idx}_best_model.pth")

    # Define model (parameters identical to training)
    model = AFP_Spectra_Model(
        in_channels=node_feat_dim,
        hidden_dim=128,
        out_dim=spectrum_len,
        edge_dim=edge_feat_dim,
        num_layers=3,
        num_timesteps=2,
        dropout=0.1
    ).to(device)

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode

    # Build DataLoader
    new_loader = DataLoader(new_dataset, batch_size=32)

    # Predict and evaluate EMD
    emd_list = []
    all_preds = []
    all_trues = []
    all_smiles = []

    with torch.no_grad():
        for batch in new_loader:
            batch = batch.to(device)
            preds = model(batch)
            trues = batch.y

            # Record EMD for each sample
            for i in range(batch.num_graphs):
                sample_emd = emd_loss(preds[i:i+1], trues[i:i+1]).item()
                emd_list.append(sample_emd)

            # Save predictions, ground truth, and SMILES
            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(trues.cpu().numpy())
            all_smiles.extend(batch.smiles)

    # Compute average EMD for this fold
    avg_emd = statistics.mean(emd_list) if emd_list else 0.0
    std_emd = statistics.stdev(emd_list) if len(emd_list) > 1 else 0.0
    all_fold_emd.append(avg_emd)

    print(f"Fold {fold_idx} EMD on new dataset: {avg_emd:.5f}")

    # Save prediction results for this fold
    fold_pred_path = f"./Fold_Predictions/4.0_50_100_Fold_{fold_idx}_predictions.pickle"

    # Automatically create the output directory if it does not exist
    os.makedirs(os.path.dirname(fold_pred_path), exist_ok=True)

    with open(fold_pred_path, "wb") as f:
        pickle.dump({
            "smiles": all_smiles,
            "predictions": all_preds,
            "true_spectra": all_trues,
            "emd_list": emd_list
        }, f)

    print(f"Fold {fold_idx} prediction results saved to {fold_pred_path}")

# Compute the average EMD across all folds
avg_all_fold_emd = statistics.mean(all_fold_emd)
print(f"\nAverage EMD across all folds: {avg_all_fold_emd:.5f}")
