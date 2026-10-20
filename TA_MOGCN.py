from library import combine_module2
import pickle
import time
import random
import copy
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import torch
import warnings


import torch.nn.functional as F
from torch_geometric.utils import dropout_adj

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


base_path = ''
dataset_name = 'sample_data'
data_path_node = base_path + 'data/' + dataset_name + '/'


base = ['mRNA', 'DNA', 'miRNA']
new_base = ['n_mRNA', 'n_DNA', 'n_miRNA']
netw_base = ['1_', '2_', '3_']

max_epochs = 1000
min_epochs = 300
patience = 80
random_state = 897
learning_rate = 0.005
hd_size = 32
log_interval = 25

print('ROSMAP Single-Run Training Demo started.')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, optimizer, dynamic_drop_p, dynamic_alpha):
    model.train()
    optimizer.zero_grad()


    if dynamic_drop_p > 0:
        ei1, ea1 = dropout_adj(DATA[0].edge_index, DATA[0].edge_attr, p=dynamic_drop_p, training=True)
        ei2, ea2 = dropout_adj(DATA[1].edge_index, DATA[1].edge_attr, p=dynamic_drop_p, training=True)
        ei3, ea3 = dropout_adj(DATA[2].edge_index, DATA[2].edge_attr, p=dynamic_drop_p, training=True)

        d1 = copy.copy(DATA[0])
        d1.edge_index = ei1
        d1.edge_attr = ea1

        d2 = copy.copy(DATA[1])
        d2.edge_index = ei2
        d2.edge_attr = ea2

        d3 = copy.copy(DATA[2])
        d3.edge_index = ei3
        d3.edge_attr = ea3
    else:
        d1, d2, d3 = DATA[0], DATA[1], DATA[2]

    out1, coef1, coef2, coef3 = model(d1, d2, d3)
    mask = DATA[0].train_mask
    targets = DATA[0].y[mask]

    loss_main = criterion(out1[mask], targets)


    if dynamic_alpha > 0:
        x_emb1 = F.relu(model.conv1(d1.x, d1.edge_index, d1.edge_attr))
        x_emb2 = F.relu(model.conv2(d2.x, d2.edge_index, d2.edge_attr))
        x_emb3 = F.relu(model.conv3(d3.x, d3.edge_index, d3.edge_attr))

        logits1 = model.conv11(x_emb1, d1.edge_index, d1.edge_attr)
        logits2 = model.conv22(x_emb2, d2.edge_index, d2.edge_attr)
        logits3 = model.conv33(x_emb3, d3.edge_index, d3.edge_attr)

        loss_aux1 = criterion(logits1[mask], targets)
        loss_aux2 = criterion(logits2[mask], targets)
        loss_aux3 = criterion(logits3[mask], targets)

        w1 = coef1[mask].mean().item()
        w2 = coef2[mask].mean().item()
        w3 = coef3[mask].mean().item()

        total_loss = loss_main + dynamic_alpha * (w1 * loss_aux1 + w2 * loss_aux2 + w3 * loss_aux3)
    else:
        total_loss = loss_main

    total_loss.backward()
    optimizer.step()

    return out1, total_loss


def validate(model):
    model.eval()
    with torch.no_grad():
        out1, _, _, _ = model(DATA[0], DATA[1], DATA[2])
        loss1 = criterion(out1[DATA[0].valid_mask], DATA[0].y[DATA[0].valid_mask])
    return loss1


DF = {}
for i in range(len(base)):
    with open(base_path + 'dataset/' + netw_base[i] + '.csv') as f:
        df_ = pd.read_csv(f, header=None)
    DF[i] = torch.tensor(df_.values, device=device, dtype=torch.float32)


EDGES = {}
for n in range(int(len(new_base))):
    with open(data_path_node + 'edges_' + new_base[n] + '.pkl', 'rb') as f:
        edge_index = pickle.load(f)
    EDGES[n] = torch.tensor(edge_index, device=device)


labels = pd.read_csv('dataset/labels_.csv', header=None).to_numpy()[:, 0]
labels = labels.astype(int)

unexpected_classes = sorted(set(np.unique(labels).astype(int).tolist()) - {0, 1})
if unexpected_classes:
    raise ValueError(
        f'labels_.csv 中存在非 0/1 的 ROSMAP 标签: {unexpected_classes}。'
        '请核对标签编码。'
    )


torch.manual_seed(random_state)
np.random.seed(random_state)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_state)

num_nodes = DF[0].shape[0]
num_edges = EDGES[0].shape[1]
avg_degree = num_edges / num_nodes

degree_factor = max(0.0, min(1.0, (avg_degree - 3.5) / 1.5))

criterion = torch.nn.CrossEntropyLoss()

dynamic_wd = 1e-4 * degree_factor
dynamic_drop_p = 0.20 * degree_factor
dynamic_alpha = 0.30 * degree_factor

print(f"Device: {device} | Seed: {random_state} | LR: {learning_rate} | Hidden: {hd_size}")
print(f"Graph Avg Degree: {avg_degree:.2f} | Topology Factor (lambda): {degree_factor:.2f}")
print(f"Config -> DropEdge: {dynamic_drop_p:.3f} | Aux: {dynamic_alpha:.3f} | L2: {dynamic_wd:.5f}")

start = time.time()
alltrain_idx, test_idx = train_test_split(
    np.arange(len(labels)),
    test_size=0.2,
    shuffle=True,
    stratify=labels,
    random_state=random_state
)
train_idx, val_idx = train_test_split(
    alltrain_idx,
    test_size=0.25,
    shuffle=True,
    stratify=labels[alltrain_idx],
    random_state=random_state
)

DATA = {}
for i in range(len(base)):
    dim1, dim2 = EDGES[i].shape
    DATA[i] = Data(
        x=DF[i],
        edge_index=EDGES[i],
        edge_attr=torch.tensor(np.ones((dim2)), device=device, dtype=torch.float32),
        y=torch.tensor(labels, device=device).long()
    )

in_size1 = DATA[0].x.shape[1]
in_size2 = DATA[1].x.shape[1]
in_size3 = DATA[2].x.shape[1]
out_size = torch.unique(DATA[0].y).shape[0]

DATA[0].train_mask = torch.tensor(
    np.array([i in set(train_idx) for i in range(DATA[0].x.shape[0])]),
    device=device
)
DATA[0].valid_mask = torch.tensor(
    np.array([i in set(val_idx) for i in range(DATA[0].x.shape[0])]),
    device=device
)

print(f"Samples: train={len(train_idx)}, validation={len(val_idx)}, test={len(test_idx)}")

model = combine_module2.GCN(
    in_size1=in_size1,
    in_size2=in_size2,
    in_size3=in_size3,
    hid_size1=hd_size,
    out_size=out_size
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=dynamic_wd
)

min_valid_loss = np.inf
patience_count = 0
best_model_wts = copy.deepcopy(model.state_dict())
best_epoch = 0

print('Training started (one run; fixed demonstration hyperparameters).')
for epoch in range(max_epochs):
    out, tr_loss = train(model, optimizer, dynamic_drop_p, dynamic_alpha)
    this_valid_loss = validate(model)

    if this_valid_loss < min_valid_loss:
        min_valid_loss = this_valid_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        best_epoch = epoch + 1
        patience_count = 0
    else:
        patience_count += 1

    if (epoch + 1) % log_interval == 0 or epoch == 0:
        print(
            f'Epoch {epoch + 1:4d}/{max_epochs} | '
            f'Train Loss: {tr_loss.item():.4f} | '
            f'Val Loss: {this_valid_loss.item():.4f} | '
            f'Best Val Loss: {min_valid_loss.item():.4f}'
        )

    if (epoch + 1) >= min_epochs and patience_count >= patience:
        print(f'Early stopping at epoch {epoch + 1}.')
        break

model.load_state_dict(best_model_wts)

valid_mask = np.array([i in set(test_idx) for i in range(DATA[0].x.shape[0])])

model.eval()
with torch.no_grad():
    out1_v, _, _, _ = model(DATA[0], DATA[1], DATA[2])
    predictions = out1_v.argmax(dim=1)

y_test = labels[valid_mask]
pred_ = predictions[valid_mask].cpu().numpy()

y_test_int = y_test.astype(int)
pred_int = pred_.astype(int)

acc_i = accuracy_score(y_test_int, pred_int)
wf1_i = f1_score(y_test_int, pred_int, average='weighted')
mf1_i = f1_score(y_test_int, pred_int, average='macro')
mcc_i = matthews_corrcoef(y_test_int, pred_int)

print('\n=== ROSMAP Single-Run Demo Results ===')
print(f'Completed Epochs: {epoch + 1} | Best Epoch: {best_epoch}')
print(f'Best Validation Loss: {min_valid_loss.item():.4f}')
print(f'Accuracy:    {acc_i:.3f}')
print(f'Weighted-F1: {wf1_i:.3f}')
print(f'Macro-F1:    {mf1_i:.3f}')
print(f'MCC:         {mcc_i:.3f}')
print(f'Runtime (s): {time.time() - start:.1f}')
