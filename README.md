# Deep Learning brathing-induced artifact correction for accelerated MRI

Hi! 

This project was developed during the Brainhack School 2026 at Polytechnique Montréal.

<a href="https://github.com/Annaelle8">
  <img src="https://avatars.githubusercontent.com/u/128856434?v=4&size=64" width="100px;" alt=""/>
  <br /><sub><b>Annaelle8</b></sub>
</a>


---

## Overview

Respiratory motion during MRI acquisition introduces artifacts in k-space that degrade image quality. This is particularly critical for spinal cord imaging. 
This project aim to simulate breathing-induced motion corruption directly in k-space and trains a 2D U-Net to correct these artifacts. Moreover this project aim to simulate accelerated MRI
that's are more and more used because MRI has a long acquisition time, so un undersampling factor is also add in k-space as well as some complex Gaussian noise.


Rather than working in image space, the model operates on **complex k-space data** (real + imaginary channels),
 which is more faithful to the actual acquisition process and allows correction before reconstruction.

---

## Scientific Background

During MRI acquisition, k-space lines are acquired sequentially over time. Respiratory motion between acquisitions induces a **phase shift** in each k-space line, modeled as:

$$\tilde{K}(k_x, k_y) = K(k_x, k_y) \cdot e^{-j2\pi k_x \cdot d(k_y)}$$

where $d(k_y)$ is the respiratory displacement at the time of acquisition of line $k_y$.

Combined with **Cartesian undersampling** (acceleration factor R) and **Gaussian noise**, this produces realistic corrupted k-space data used for supervised training.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT                                     │
│              Clean T2w NIfTI volume (ds005616)                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │  2D FFT
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CLEAN K-SPACE                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
        Phase ramp    Cartesian      Complex
        (motion)    undersampling    Gaussian
          A, f_r        (R)          noise
                                     (SNR)
              │             │             │
              └─────────────┼─────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CORRUPTED K-SPACE                              │
│              (real + imaginary channels)                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    2D U-NET                                      │
│         Input:  (2, H, W) — corrupted k-space                   │
│         Output: (2, H, W) — corrected k-space                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │  iFFT
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   RECONSTRUCTED IMAGE                            │
└─────────────────────────────────────────────────────────────────┘
```

### Corruption Parameters

| Parameter | Range | Description |
|---|---|---|
| A | 2.0, 5.0, 8.0 px | Motion amplitude |
| f | 12, 15, 18 breaths/min | Respiratory rate |
| R | 2, 4, 6 | Undersampling factor |
| SNR | 15, 20, 25 dB | Signal-to-noise ratio |

Total combinations: **81 per slice** → 231,417 training samples across 56 subjects.

---

## Dataset

This project uses the **Whole-Spine Anatomical MRI dataset** (ds005616), available on OpenNeuro:

> [https://openneuro.org/datasets/ds005616/versions/1.1.2](https://openneuro.org/datasets/ds005616/versions/1.1.2)

- **Modality**: T2-weighted sagittal whole-spine MRI
- **Subjects**: 56
- **Resolution**: varies across subjects (227–424 × 599–729 px)
- **Format**: NIfTI (.nii.gz), BIDS-compliant

### Data access via Datalad

```bash
datalad install https://github.com/OpenNeuroDatasets/ds005616.git
datalad get sub-*/anat/sub-*_T2w.nii.gz
```

---

## Repository Structure

```
Sarrazin_project/
│
├── ds005616/                          # Dataset (not tracked by git)
│
├── src/
│   ├── Kspace_simulation.py           # K-space corruption pipeline
│   ├── Utils.py                       # Metrics and utilities
│   ├── Unet_model.py                  # 2D U-Net architecture
│   └── Unet_train.py                  # Training loop (online generation)
│
├── notebooks/
│   ├── Kspace_corruption_simulation_vf.ipynb   # Step-by-step simulation
│   ├── Unet_inference.ipynb                    # Inference & visualization
│   └── Unet_analysis.ipynb                     # Training curves & metrics
│
├── training_data/
│   ├── manit_v2.csv                # Dataset manifest (path, TR, TE, H, W, params)
│   └── splits.json                    # Train/val/test subject splits (70/15/15)
│
├── results/
│   ├── fulln_v2/
│   │   ├── unet_best.pt               # Best model checkpoint
│   │   ├── training_history.csv       # Metrics per epoch
│   │   └── checkpoint_epoch*.pt       # Periodic checkpoints
│   └── figures/                       # Generated figures
│
├── train_full.sh                      # SLURM job script (Alliance Canada)
└── README.md
```

---

## Installation

### Requirements

- Python 3.10
- PyTorch ≥ 2.0
- CUDA (for GPU training)

### Setup

```bash
# Clone the repository
git clone https://github.com/annaellesarrazin/Sarrazin_project.git
cd Sarrazin_project

# Create conda environment
conda create -n brainhack python=3.10
conda activate brainhack

# Install dependencies
pip install torch torchvision
pip install nibabel numpy pandas matplotlib scikit-image
pip install neurokit2 scipy pybids joblib tqdm
```

---

## Usage

### 1. Generate the manifest

Run the manifest generation cells in `notebooks/Kspace_corruption_simulation_vf.ipynb`.

This creates `training_data/manifest_v2.csv` with one row per (subject × slice × corruption combo).

### 2. Train the model

**Local test (2 subjects, 1 epoch):**

```bash
python src/Unet_train.py \
    --data_root /path/to/Sarrazin_project \
    --manifest  training_data/manifest_small.csv \
    --splits    training_data/splits.json \
    --output    results/test_run \
    --epochs    1 \
    --batch_size 4
```

**Alliance Canada (Narval):**

```bash
sbatch train_full.sh
```

Monitor training:

```bash
tail -f logs/unet_*.out
```

### 3. Inference & visualization

Open `notebooks/Unet_inference.ipynb` and set:

```python
MODEL_PATH  = Path('results/full_run/unet_best.pt')
SPLITS_PATH = Path('training_data/splits.json')
```

Run all cells to visualize GT | Corrupted | Reconstructed for validation subjects.

### 4. Analyze training

Open `notebooks/Unet_analysis.ipynb` and set:

```python
HISTORY_PATH = Path('results/full_run/training_history.csv')
```

---

## Model Architecture

**2D U-Net** with 3 pooling levels:

```
Input (2, H, W) — real & imaginary corrupted k-space
    │
    ├── Encoder: 32 → 64 → 128 → 256 (bottleneck)
    │   MaxPool2d between levels
    │
    ├── Decoder: 256 → 128 → 64 → 32
    │   ConvTranspose2d + skip connections
    │
Output (2, H, W) — real & imaginary corrected k-space
```

- **Parameters**: ~1.9M
- **Loss**: L1 (less blurry than MSE)
- **Optimizer**: Adam (lr=1e-3)
- **Scheduler**: ReduceLROnPlateau (patience=5, factor=0.5)
- **Input normalization**: divided by max(|clean k-space|)

---

## Limitations

- Motion model is **1D rigid translation** only (no rotation, no non-rigid deformation)

---

## References


---
## Author

**Annaelle Sarrazin** — Brainhack School 2026
