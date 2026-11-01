# TA-MOGCN: ROSMAP Single-Run Training Demo

## 1. Overview

This repository provides a single-run demonstration of TA-MOGCN for
binary classification on ROSMAP. It illustrates the complete workflow,
including loading preprocessed multi-omics data, constructing sample
similarity graphs, training the model, selecting the best model based on
validation loss, applying early stopping, and evaluating the selected
model on the test set.

The demonstration uses a fixed random seed and a fixed hyperparameter
configuration to show one complete training and evaluation run. The
ROSMAP dataset and preprocessed input files are not distributed with
this repository; users must prepare the required files before running
the code.

## 2. Method Overview

TA-MOGCN combines a multi-omics graph-learning architecture with
patient-level modality attention and a topology-factor-driven adaptive
training mechanism. The demonstration includes the following steps:

1.  Load the three preprocessed omics feature matrices (mRNA, DNA
    methylation, and miRNA) and the binary labels.
2.  Construct one sample similarity graph for each omics modality using
    cosine distance and a modality-specific threshold selected from
    pairwise sample distances.
3.  Learn modality-specific sample representations through three GCN
    branches.
4.  Apply patient-level modality attention to assign modality weights to
    each patient and fuse the modality-specific representations. The
    fused representation is passed through three classification heads,
    whose outputs are summed.
5.  Calculate a topology factor, λ, from the stored edge-index entries
    of the mRNA reference graph per node, including self-loops. Use this
    factor to scale the DropEdge probability, auxiliary-loss
    coefficient, and weight-decay coefficient.
6.  Optimize the model using training labels, monitor validation loss
    for model selection and early stopping, and restore the model
    parameters associated with the lowest validation loss.
7.  Evaluate the restored model on the test set and report Accuracy,
    Weighted-F1, Macro-F1, and MCC.

The demo uses the following fixed settings:

| Parameter                                         |           Value |
|---------------------------------------------------|----------------:|
| Random seed                                       |             897 |
| Learning rate                                     |           0.005 |
| Hidden dimension                                  |              32 |
| Model dropout                                     |             0.5 |
| Graph-construction sparsity parameter             |               5 |
| Maximum DropEdge coefficient                      |        0.20 × λ |
| Auxiliary-loss coefficient                        |        0.30 × λ |
| Weight-decay coefficient                          |    1 × 10⁻⁴ × λ |
| Maximum epochs                                    |            1000 |
| Earliest epoch at which early stopping is allowed |             300 |
| Early-stopping patience                           |              80 |
| Logging interval                                  | Every 25 epochs |
| Number of runs                                    |               1 |

The topology factor and the resulting training coefficients are
calculated once before training and remain fixed throughout this single
run.

## 3. Dependencies

The code depends on the following Python packages:

- Python
- PyTorch
- PyTorch Geometric
- NumPy
- pandas
- SciPy
- scikit-learn
- openpyxl

Install versions of PyTorch and PyTorch Geometric that are compatible
with your operating system, Python version, and CUDA environment. The
training script uses `torch_geometric.utils.dropout_adj`. This API is
deprecated in some PyTorch Geometric releases, so use a compatible
version that still provides it.

## 4. Input Files and Graph Construction

Run all commands from the repository root because the scripts use
relative paths.

### 4.1 Input Files

The `dataset/` directory must contain the following files:

| File          | Description                            |
|---------------|----------------------------------------|
| `1_.csv`      | Preprocessed mRNA expression features  |
| `2_.csv`      | Preprocessed DNA methylation features  |
| `3_.csv`      | Preprocessed miRNA expression features |
| `labels_.csv` | Binary class labels (0 and 1)          |

The ROSMAP dataset and these input files are not included in this
repository. Users must prepare the preprocessed input files separately.

Each feature matrix must contain samples in rows and features in
columns. The CSV files must not contain a header row or a sample-ID
column, and all feature values must be finite numeric values. The label
file must contain exactly one column without a header. Labels must be
encoded as integers `0` and `1`, and both classes must be present.

All three feature matrices and the label file must contain the same
number of samples and follow exactly the same sample order.

These files are expected to be preprocessed input matrices. This
demonstration script does not perform the preprocessing steps described
in the paper, including variance-based feature filtering.

### 4.2 Construct Sample Similarity Graphs

After preparing `dataset/1_.csv`, `dataset/2_.csv`, and
`dataset/3_.csv`, create the graph output directory and run `main.py`.

On Windows Command Prompt:

```bat
if not exist data\sample_data mkdir data\sample_data
python main.py
```

On Linux or macOS:

```bash
mkdir -p data/sample_data
python main.py
```

`main.py` uses the cosine-distance and adjacency-construction functions
in `utils.py`. The graph-construction sparsity parameter is set to `5`
in the script (`thershold = 5`). For each omics modality, a
cosine-distance threshold is selected from the sorted pairwise sample
distances using this parameter. The parameter controls the global
threshold-selection procedure; it does not enforce exactly five
neighbors for every sample.

The script generates the following edge-index files:

- `data/sample_data/edges_n_mRNA.pkl`
- `data/sample_data/edges_n_DNA.pkl`
- `data/sample_data/edges_n_miRNA.pkl`

The graph-construction function adds self-loops before returning the
sparse adjacency representation. The saved files retain edge indices,
including stored self-loop entries, rather than normalized adjacency
values. During training, unit edge weights are assigned to these edge
indices, and `GCNConv` performs graph normalization.

### 4.3 Topology Factor

Before training, `TA_MOGCN.py` calculates the mRNA reference graph's
average degree using the number of stored edge-index entries divided by
the number of nodes. Self-loop entries are included in this count.

The implementation uses the following calculations:

```python
num_nodes = DF[0].shape[0]
num_edges = EDGES[0].shape[1]
avg_degree = num_edges / num_nodes
degree_factor = max(0.0, min(1.0, (avg_degree - 3.5) / 1.5))

dynamic_drop_p = 0.20 * degree_factor
dynamic_alpha = 0.30 * degree_factor
dynamic_wd = 1e-4 * degree_factor
```

The topology factor is computed once before training. Its value and the
resulting coefficients remain unchanged during the run.

## 5. Run the Single-Run Demo

After preparing the input files and generating the graph files, run the
following command from the repository root:

```bash
python TA_MOGCN.py
```

The script performs one training run and periodically prints the
training loss, validation loss, and best validation loss to the
terminal. After training, it restores the model parameters associated
with the lowest validation loss and evaluates that model on the test
set.

The script reports the completed epoch, best epoch, best validation
loss, test Accuracy, Weighted-F1, Macro-F1, MCC, and runtime. The
selected model parameters are restored in memory for evaluation; the
script does not save a model checkpoint or generate result figures.

## 6. Data Split and Evaluation Setting

The code uses stratified random splitting. It first holds out 20% of the
samples for testing and then holds out 25% of the remaining 80% for
validation. The target proportions are 60% training, 20% validation, and
20% testing. Exact sample counts may vary because of stratified
splitting and rounding.

This demonstration uses a transductive graph-learning setting. The
sample similarity graphs are constructed from all sample features before
the data split, and graph message passing uses the full set of sample
nodes. Only training labels are used to calculate the training loss.
Validation labels are used for model selection and early stopping, while
test labels are used for final evaluation. Thus, test-sample features
and graph connections participate in graph propagation, but test labels
are not used for training or model selection.

## 7. Relationship to the Paper's Experiments

This repository provides a single-run demonstration on ROSMAP using a
fixed random seed and hyperparameter configuration. Its purpose is to
illustrate the TA-MOGCN graph-construction, training, validation, and
testing workflow.

The classification results reported in the paper were obtained using the
repeated multi-seed evaluation protocol described in the manuscript. The
single-run demonstration in this repository is not intended to reproduce
the paper's aggregated results by itself.

## 8. Code Provenance and License

The multi-omics graph-learning backbone and parts of the foundational
code in this project are based on and adapted from the [official IGCN
implementation](https://github.com/bozdaglab/IGCN). IGCN provides the
underlying multi-omics graph convolutional architecture, including
modality-specific GCN branches, patient-level modality attention,
attention-weighted feature fusion, and multiple classification heads.
TA-MOGCN uses this architecture as its representation-learning and
fusion backbone and introduces the proposed topology-factor-driven
adaptive training mechanism to regulate training constraints according
to the connectivity of the mRNA reference graph.

The roles of the individual code files are as follows:

- `library/combine_module2.py`: Implements the IGCN-derived multi-omics
  graph convolutional backbone, including three GCN branches,
  patient-level modality attention, attention-weighted feature fusion,
  and three classification heads whose outputs are summed. This backbone
  provides the representation-learning and fusion architecture used by
  TA-MOGCN.
- `utils.py`: Adapts IGCN functions for cosine-distance calculation,
  distance-threshold-based sample similarity graph construction, and
  sparse adjacency processing. The adjacency symmetrization and
  normalization implementation has been modified for this project. In
  the current pipeline, the graph files retain edge indices; unit edge
  weights are assigned during training, and `GCNConv` performs graph
  normalization.
- `main.py`: Adapts IGCN's multi-omics sample similarity
  graph-construction workflow. It reads the three omics feature matrices
  separately, constructs the corresponding sample similarity graphs
  using the sparsity parameter specified in the script, and generates
  the associated edge-index files.
- `TA_MOGCN.py`: Implements the TA-MOGCN training pipeline and the
  proposed topology-adaptive training mechanism. The mechanism
  calculates a topology factor from the mRNA reference graph and uses it
  to regulate the DropEdge probability, auxiliary-loss coefficient, and
  weight-decay coefficient. The script provides the single-run ROSMAP
  training and evaluation demonstration.

The current `LICENSE` in the official IGCN GitHub repository specifies
the **Creative Commons Attribution-NonCommercial 4.0 International (CC
BY-NC 4.0)** license. When sharing adapted material, retain attribution
and the [license
link](https://github.com/bozdaglab/IGCN/blob/main/LICENSE), identify the
modifications, and comply with the non-commercial restriction. This
repository is intended for non-commercial academic research and
educational use. This notice does not grant commercial-use permission
for IGCN-derived components. For commercial use, obtain appropriate
authorization from the relevant rights holders.

This project is not affiliated with, endorsed by, or an official
collaboration with the IGCN authors or their team. The code is provided
as-is, without warranties regarding its suitability, error-free
operation, or fitness for a particular purpose.

## 9. Citation

If this repository is useful in your research, please cite the
corresponding TA-MOGCN paper. When using or discussing the IGCN-derived
architecture or adapted implementation, also cite the IGCN paper:

Ozdemir C, Vashishath Y, Bozdag S, Alzheimer's Disease Neuroimaging
Initiative. IGCN: integrative graph convolution networks for patient
level insights and biomarker discovery in multi-omics integration.
*Bioinformatics*. 2025;41(6):btaf313.
<https://doi.org/10.1093/bioinformatics/btaf313>.
