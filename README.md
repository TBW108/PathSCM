# PathSCM: reproducible prediction benchmarks

This repository reproduces the current **PathSCM, GEARS, scGen, discrepancy-VAE, and SENA** prediction benchmark on **K562, RPE1, and HepG2**. It contains model code, training configurations, data preparation, numerical evaluation, and tests. It does not include manuscript files or plotting scripts.

The main workflow is **seen perturbations in held-out cell batches**, with three model seeds. A separate, audited **unseen-gene** workflow supports PathSCM and GEARS. These protocols are not interchangeable: a model fitted on a gene cannot be reused as an unseen model for that gene.

## Layout

```text
run.py                  train / evaluate / summarize
configs/                actual checkpoint-derived settings, frozen splits and annotation snapshot
pathscm/                model, SCM, shared components, training and unseen transfer
data/                   download, preparation, loading and verification
baselines/              official-method wrappers, adapters, source setup and environment requirements
evaluation/             prediction interfaces, original protocols, metrics and aggregation
tests/                  numerical tests and a two-step synthetic training check
```

`datasets/`, `checkpoints/`, `results/`, third-party source checkouts, and environments are created locally and excluded from Git. All commands below run **from this repository's root**.

## 1. Environments

Use Linux and Python **3.10** for the main environment. The three real datasets are large; preparation materializes expression matrices and evaluation fits PCA on 50,000 training cells. A CUDA GPU is recommended for complete experiments. The original scGen training wrapper requires CUDA.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# First install a PyTorch build appropriate to your CUDA/driver or CPU environment.
python -m pip install -r requirements-dev.txt
python tests/run_checks.py
```

`requirements.txt` pins the observed scientific Python package versions, while allowing hardware-appropriate PyTorch installation. `configs/validated_environment.json` records the exact existing environment used for the consistency checks, including PyTorch. Install PyTorch using its [official instructions](https://pytorch.org/get-started/locally/). Different hardware/library versions and newly trained weights need not reproduce floating-point results bit for bit.

Fetch the official baseline sources at the exact revisions used in the experiments:

```bash
python -m baselines.setup --method all
```

This clones GEARS into `baselines/GEARS`, and discrepancy-VAE/SENA into `baselines/sota_src`. Existing checkouts must already match the pinned revisions; the setup tool does not reset them.

| Method | Source | Pinned revision |
|---|---|---|
| GEARS | [snap-stanford/GEARS](https://github.com/snap-stanford/GEARS) | `f374e43e197b295016d80395d7a54ddb81cc6769` |
| discrepancy-VAE | [uhlerlab/discrepancy_vae](https://github.com/uhlerlab/discrepancy_vae) | `4451fdbc9d0aa3a1dee4e7d1b743a434e98fa58a` |
| SENA | [ML4BM-Lab/SENA](https://github.com/ML4BM-Lab/SENA) | `d5511aa31982f42995d55610f590f5891b3d3da8` |
| scGen | [theislab/scgen](https://github.com/theislab/scgen) | released package **2.1.0**, scvi-tools **0.18.0** |

The two CRL methods use the main environment and their official training loops. Their adapter disables optional online logging. GEARS uses its own scientific Python versions:

```bash
python3.10 -m venv .venv-gears
.venv-gears/bin/python -m pip install -r baselines/requirements/gears.txt
# The wrapper imports the pinned local GEARS checkout directly.
export PATHSCM_GEARS_PYTHON="$PWD/.venv-gears/bin/python"
```

scGen uses its original **Python 3.9** environment, separately from PathSCM. Lightning 1.7.7 has legacy dependency metadata, so use pip 24.0 for installation:

```bash
conda create -n pathscm-scgen python=3.9 pip=24.0 -y
conda run -n pathscm-scgen python -m pip install 'setuptools<81'
conda run -n pathscm-scgen python -m pip install -r baselines/requirements/scgen.txt
conda run -n pathscm-scgen python -c 'import scgen, scvi; print(scgen.__version__, scvi.__version__)'
export PATHSCM_SCGEN_PYTHON="$(conda run -n pathscm-scgen python -c 'import sys; print(sys.executable)')"
```

The scGen requirements record the packages observed in the working training environment. Fresh environment installation and full third-party retraining were not performed during repository preparation; existing baseline environments and prediction adapters were checked. Consult each upstream repository for its license and hardware-specific installation requirements.

## 2. Obtain the three datasets

| Dataset | Original input used here | Source |
|---|---|---|
| K562 essential, Replogle 2022 | scPerturb `ReplogleWeissman2022_K562_essential.h5ad`, 310,385 cells × 8,563 genes | [scPerturb record 7416068](https://zenodo.org/records/7416068) |
| RPE1 essential, Replogle 2022 | GEARS-processed `perturb_processed.h5ad`, log-normalized, 162,733 cells × 5,000 genes | [Harvard Dataverse datafile 7458694](https://dataverse.harvard.edu/api/access/datafile/7458694) |
| HepG2 essential, Nadig 2025 | `GSE264667_hepg2_raw_singlecell_01.h5ad`, 145,473 cells × 9,624 genes | [author-hosted dataset](https://huggingface.co/datasets/arcinstitute/Replogle-Nadig-Preprint/blob/main/GSE264667_hepg2_raw_singlecell_01.h5ad), [GEO GSM8225195](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8225195) |

```bash
python -m data.download --dataset k562
python -m data.download --dataset rpe1
python -m data.download --dataset hepg2
unzip -n datasets/raw/replogle_rpe1_essential.zip -d datasets/raw/rpe1_archive
find datasets/raw/rpe1_archive -name perturb_processed.h5ad
```

Use the `perturb_processed.h5ad` path printed by the last command below. The downloader refuses to overwrite an existing file and writes to a temporary `.download` file until complete. It does not silently substitute a different dataset version.

For pathway annotations, **use the supplied snapshot** in `configs/pathways/`: `ReactomePathways.gmt` and `ReactomePathwaysRelation.txt`. These files were copied from the original experiment, with original Reactome pathway identifiers preserved. Their source is [Reactome downloads](https://reactome.org/download-data). Replacing them with the latest database may change the genes/pathways and invalidate comparison with the saved reference metrics.

## 3. Prepare the seen benchmark

```bash
python -m data.prepare --dataset k562 --input datasets/raw/replogle_2022_k562_essential.h5ad
python -m data.prepare --dataset rpe1 --input datasets/raw/rpe1_archive/replogle_rpe1_essential/perturb_processed.h5ad
python -m data.prepare --dataset hepg2 --input datasets/raw/hepg2_raw_singlecell_01.h5ad
```

Adjust only the RPE1 archive-internal path if the extraction layout differs. RPE1 is explicitly treated as already log-normalized; K562 and HepG2 are normalized from counts. If using an existing converted RPE1/HepG2 input containing `perturbation`, `batch`, and `guide_id`, pass `--converted`.

The unchanged preparation algorithm first splits entire gem groups (8 validation, 8 test, remaining training, seed 0), fits 2,000 highly variable genes and scaling on training cells, excludes top-level Reactome pathways, and deduplicates pathway memberships. It prepares the full eligible pathway set first and then selects the 64 nodes with the most perturbed members. **Do not replace this two-step process with direct 64-node preprocessing for the seen benchmark**: the reference cells and scaling must remain the same.

Outputs, for each `DATASET` in `k562 rpe1 hepg2`:

```text
datasets/processed/DATASET_generative_v3.h5ad       full-node reference / GEARS input
datasets/processed/DATASET_generative_v3_k64.h5ad   PathSCM, scGen, dVAE, SENA input
```

Preparation refuses to overwrite existing outputs and checks the genes, cell identities, cell splits, condition names, node masks, pathway supports, and scaling against frozen fingerprints from the original experiment. To check existing prepared files:

```bash
python -m data.verify --dataset k562
python -m data.verify --dataset rpe1
python -m data.verify --dataset hepg2
```

## 4. Train all five methods

Inspect commands first; this does not start training:

```bash
python run.py train --dataset k562 --method all --seeds 0 1 2 --dry-run
```

Run each dataset:

```bash
python run.py train --dataset k562 --method all --seeds 0 1 2 --device cuda
python run.py train --dataset rpe1 --method all --seeds 0 1 2 --device cuda
python run.py train --dataset hepg2 --method all --seeds 0 1 2 --device cuda
```

Select a single method with `--method PathSCM`, `GEARS`, `scGen`, `discrepancy-VAE`, or `SENA`. Jobs run sequentially in the foreground. The launcher refuses to overwrite an existing model/output. Reuse completed models by skipping their training and proceeding to evaluation; select missing methods/seeds explicitly.

PathSCM uses the actual saved arguments in `configs/{k562,rpe1,hepg2}.json`: shared encoder, 64 factors, 16 background dimensions, SCM likelihood, energy-distance generation loss, 13,000 joint training steps, 36 conditions and 128 cells per condition per step. The low-level trainer retains options required by the shared model implementation; do not use its historical defaults in place of these configurations.

GEARS uses 20 epochs, batch size 256, hidden size 64, learning rate 0.001 and its original target split. scGen uses its original architecture and 100-epoch maximum, batch size 32, patience 25. discrepancy-VAE and SENA retain their respective official defaults with latent dimension 64. The same prepared cells and original adapters are used; baseline architectures are not reimplemented as substitutes.

GEARS downloads its GO/essential-gene priors into `baselines/gears_data/` on first use. Internet access is needed then; later runs reuse those files.

## 5. Evaluate and output metrics

Run evaluation in the **main environment**, including scGen exports. It uses the original prediction adapters without importing scGen's training environment:

```bash
python run.py evaluate --dataset k562 --method all --seeds 0 1 2 --device cuda
python run.py evaluate --dataset rpe1 --method all --seeds 0 1 2 --device cuda
python run.py evaluate --dataset hepg2 --method all --seeds 0 1 2 --device cuda
python run.py summarize --protocol seen
```

For a limited inference check with an existing checkpoint:

```bash
python run.py evaluate --dataset k562 --method PathSCM --seeds 0 --device cpu --limit 2
```

`--limit` limits perturbations, not data loading or PCA preparation; it still uses the original 50,000-cell PCA. Its results go to `results/smoke/`, never the final summary. Full results go to `results/seen/`:

```text
DATASET_METHOD_seedN.csv   individual per-perturbation/per-seed metrics
per_perturbation.csv       combined observations
per_seed.csv              arithmetic mean over perturbations for each seed
summary.csv               mean and sample SD (ddof=1) across seeds
```

Final summaries require all three seeds for each included dataset/method and identical target sets across seeds. They exclude prior summary files. They summarize the methods actually evaluated; to obtain the full benchmark run all three datasets and `--method all` above.

### Exact seen metric protocol

- Frozen common evaluation sets: **800 K562, 429 RPE1, 236 HepG2** perturbations, each with at least 10 test cells. These are the original common sets, including GEARS training coverage, saved in `configs/splits/*_seen.json`.
- Expression is on the original **training-standardized log1p scale** over all 2,000 genes.
- Real effect = mean of **all** real test cells for the perturbation − mean of **all** training controls in the full-node reference data.
- Predicted effect = mean of 1,000 generated perturbation cells (seed `10000 + condition_index`) − mean of 1,000 generated control cells (seed `10000`).
- **Pearson Δ**: Pearson correlation across the 2,000 entries of those effect vectors.
- **MSE**: mean squared effect error across **all 2,000 genes**, not DE20.
- **MMD²**: the unchanged biased Gaussian-kernel estimator, summed over bandwidth scales 0.5, 1, and 2 times the median squared cross-distance. Compare the first at most 256 test cells to the same number of generated cells (seed `condition_index`), in 50 PCs fitted by Torch SVD on the original 50,000 training-cell sample (`numpy.default_rng(0)`).
- GEARS's distribution adapter shifts training control cells by its native mean prediction; MMD does not imply GEARS itself is a stochastic cell generator.

`configs/*_reference_metrics.json` contains the existing seed-averaged benchmark values for comparison. These are reference values, not results recomputed during a new run. The evaluator always computes metrics from predictions.

## 6. Optional: strict unseen-gene benchmark

This is a separate experiment for **PathSCM and GEARS only**. It uses the frozen seen/tuning/test gene identities in `configs/splits/unseen/`, fits preprocessing only on training controls and seen training genes, and selects checkpoints on tuning genes. Unseen preprocessing directly selects 64 nodes, matching the original unseen implementation.

First ensure the converted full inputs exist:

```text
datasets/raw/replogle_2022_k562_essential.h5ad
datasets/raw/rpe1_full.h5ad
datasets/raw/hepg2_full.h5ad
```

The seen preparation commands produce the latter two conversions. Also obtain GEARS's priors by running its seen pipeline, or using its official `PertData` setup. Both `gene2go_all.pkl` and `essential_all_data_pert_genes.pkl` must be present in `baselines/gears_data/`.

```bash
python run.py prepare-unseen
python run.py train --protocol unseen --dataset k562 --method all --seeds 0 1 2 --device cuda
python run.py evaluate --protocol unseen --dataset k562 --seeds 0 1 2 --device cuda
# Repeat train/evaluate for --dataset rpe1 and --dataset hepg2.
python run.py summarize --protocol unseen
```

Unseen evaluation evaluates the paired PathSCM/GEARS results together. It retains the original 2,048-cell effect prediction, seen-training-only PCA (20,000-cell maximum, sklearn randomized PCA seed 0), and randomly sampled real distribution cells. These differ intentionally from the seen protocol. Output is in `results/unseen/results/`. The transfer adapter rejects a checkpoint that was fitted on the supposed held-out genes.

## 7. Checks and scope of validation

```bash
python tests/run_checks.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python tests/smoke_train.py
```

The second command runs two optimizer steps on small synthetic data, saves/reloads a temporary checkpoint, and checks finite predictions. It is not a performance benchmark.

During packaging, 27 numerical/unit checks, nine training-configuration parsers, and the two-step training/reload check passed. Existing seed-0 checkpoints/exports were used for **15 small prediction comparisons** (3 datasets × 5 methods) against the original code. Predictions and MMD were identical. Pearson Δ and all-gene MSE were independently replayed from **45 existing effect exports** (3 datasets × 5 methods × 3 seeds); results matched the original CSVs within numerical tolerance. See `configs/validation.json` for errors and `configs/source_manifest.json` for source provenance.

These checks validate the reorganization and evaluation arithmetic. They do not assert that fresh training on different hardware produces identical weights. No complete retraining, fresh dependency installation, or full raw-data download was performed as part of packaging. Large datasets and trained weights are not included in this source folder. Current trained outputs can be evaluated by placing prepared files under `datasets/processed/` and restoring the paths defined in `evaluation/protocol.py` under `checkpoints/pathscm/`, `checkpoints/crl/`, and `checkpoints/baselines/`.
