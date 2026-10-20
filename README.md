# TA-MOGCN: ROSMAP Single-Run Training Demo

## 1. Overview

This repository provides a **Single-run Training Demo** of TA-MOGCN for binary classification on ROSMAP. It demonstrates the workflow for loading data, constructing sample similarity graphs, training the model, validating it, applying early stopping, and evaluating the selected model on a test set.

The example uses a fixed data-split seed and a fixed set of model hyperparameters to demonstrate one complete training and evaluation run.

## 2. Method Overview

The demo includes the following steps:

1. Load three omics feature matrices and binary labels.
2. Construct sample similarity graphs from the omics features and load their edge indices.
3. Learn modality-specific sample representations using three independent GCN branches.
4. Use patient-level modality attention to assign modality weights to each sample and fuse the representations.
5. Calculate a topology factor, \(\lambda\), from the average degree of the mRNA reference graph and use it to scale the DropEdge probability, auxiliary-loss coefficient, and weight-decay coefficient according to the current implementation.
6. Update model parameters using the training set, apply early stopping based on validation loss, and restore the model parameters associated with the lowest validation loss.
7. Evaluate the selected model on the test set and print Accuracy, Weighted-F1, Macro-F1, and MCC.

The demo uses the following fixed settings:

| Parameter                                         | Value           |
| ------------------------------------------------- | ---------------:|
| Random seed                                       | 897             |
| Learning rate                                     | 0.005           |
| Hidden dimension                                  | 32              |
| Maximum epochs                                    | 1000            |
| Earliest epoch at which early stopping is allowed | 300             |
| Early-stopping patience                           | 80              |
| Logging interval                                  | Every 25 epochs |
| Number of runs                                    | 1               |

These are the fixed run settings used in this example.

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

Install compatible versions of PyTorch and PyTorch Geometric for your operating system, Python version, and CUDA environment. The training script uses `torch_geometric.utils.dropout_adj`; check that the PyTorch Geometric version you install provides this API.

## 4. Input Files and Graph Construction

Run commands from the repository root because the current scripts use relative paths.

### 4.1 Input Files

The `dataset/` directory must contain:

- `1_.csv`
- `2_.csv`
- `3_.csv`
- `labels_.csv`

Labels are expected to be integers `0` and `1`. The three feature matrices, labels, and graph edge indices must refer to the same samples in exactly the same order.

### 4.2 Construct Sample Similarity Graphs

After preparing `dataset/1_.csv`, `dataset/2_.csv`, and `dataset/3_.csv`, make sure the output directory exists.

On Windows Command Prompt:

```bat
mkdir data\sample_data
python main.py
```

If the directory already exists, no further action is required. On Linux or macOS, use:

```bash
mkdir -p data/sample_data
python main.py
```

`main.py` uses the cosine-distance and adjacency-construction functions in `utils.py`. It builds graphs for the three input matrices using the threshold currently set in the script (`thershold = 5`) and generates:

- `data/sample_data/edges_n_mRNA.pkl`
- `data/sample_data/edges_n_DNA.pkl`
- `data/sample_data/edges_n_miRNA.pkl`

## 5. Run the Single-Run Demo

After preparing the input data and generating the graphs, run the following command from the repository root:

```bash
python TA_MOGCN.py
```

The script performs one training run and periodically prints the training loss, validation loss, and best validation loss to the terminal. At the end, it reports the best epoch, best validation loss, test Accuracy, Weighted-F1, Macro-F1, MCC, and runtime.

## 6. Data Split and Evaluation Setting

The code uses stratified splitting. It first holds out 20% of the samples for testing, then holds out 25% of the remaining 80% for validation. The resulting target proportions are 60% training, 20% validation, and 20% testing.

## 7. Relationship to the Paper's Experiments

This repository provides a single-run example on ROSMAP with a fixed random seed and hyperparameter configuration, illustrating the TA-MOGCN training, validation, and test workflow. The results reported in the paper were summarized according to the multi-seed evaluation protocol described in the manuscript.

## 8. Code Provenance and License

Parts of this project are based on and adapted from the official IGCN implementation:

- Original repository: [bozdaglab/IGCN](https://github.com/bozdaglab/IGCN)
- Upstream license: [IGCN LICENSE](https://github.com/bozdaglab/IGCN/blob/main/LICENSE)
- IGCN paper: [IGCN: Integrative Graph Convolution Networks for patient level insights and biomarker discovery in multi-omics integration](https://doi.org/10.1093/bioinformatics/btaf313)

The relationship between this repository and the upstream implementation is as follows:

- `library/combine_module2.py`: Retains the core IGCN structure of three GCN branches, patient-level modality attention, weighted fusion, and summed outputs from multiple classification heads. This core architecture is not claimed as a novel contribution of this project.
- `utils.py`: Adapts IGCN functions for cosine-distance calculation, distance-threshold graph construction, and sparse adjacency processing. The adjacency symmetrization and normalization implementation has been changed.
- `main.py`: Adapts IGCN's multi-omics sample-similarity graph-construction workflow. This project uses the threshold set in the current script and reads the three input matrices separately to generate the corresponding edge files.
- `train_rosmap_single_run.py`: Adapts the IGCN training workflow by adding the topology-factor-driven parameter scaling, training-time DropEdge, and auxiliary loss used in this project, and simplifying the workflow to a single-run demo.

The current `LICENSE` in the official IGCN GitHub repository specifies the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license. When sharing adapted material, retain attribution and the license link, identify the modifications, and comply with the non-commercial restriction. This repository is intended for non-commercial academic research and educational use. This notice does not grant commercial-use permission for IGCN-derived components. For commercial use, obtain appropriate authorization from the relevant rights holders.

This project is not affiliated with, endorsed by, or an official collaboration with the IGCN authors or their team. The code is provided as-is, without warranties regarding its suitability, error-free operation, or fitness for a particular purpose.

## 9. Citation

If this project is useful in your research, please cite the corresponding TA-MOGCN paper. When using or discussing the original IGCN architecture or adapted implementation, also cite the IGCN paper:

Ozdemir C, Vashishath Y, Bozdag S, Alzheimer's Disease Neuroimaging Initiative. IGCN: integrative graph convolution networks for patient level insights and biomarker discovery in multi-omics integration. *Bioinformatics*. 2025;41(6):btaf313. https://doi.org/10.1093/bioinformatics/btaf313.
