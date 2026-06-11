# Deep Learning breathing-induced artifact correction on k-space for accelerated MRI

<p align="center">
  <img src="README_figures/Main_question.png" width="700">
</p>

---
## Introduction

Hi!

This project was developed during the Brainhack School 2026 at Polytechnique Montréal.

<a href="https://github.com/Annaelle8">
  <img src="https://avatars.githubusercontent.com/u/128856434?v=4&size=64" width="100px;" alt=""/>
  <br /><sub><b>Annaelle8</b></sub>
</a>


---

## Overview

### Context

Respiratory motion during MRI acquisition introduces artifacts in k-space that degrade image quality. This is particularly critical for spinal cord imaging. This project aim to simulate breathing-induced motion corruption directly in k-space for accelerated MRI and trains a 2D U-Net to correct these artifacts. To simulate an accelerated MRI an undersempling factor ans some complex Gaussian noise have been added on the k-space.

Rather than working in image space, the model operates on **complex k-space data** (real + imaginary channels),
which is more faithful to the actual acquisition process and allows correction before reconstruction.

### Main question

Can deep learning models correct breathing motion-induced artefact on accelerated MRI from corrupted k-space acquisitions ?

### Pipeline Overview

```mermaid
flowchart TD
    A[T2w NIfTI volume: ds005616]:::input
    B[Clean K-space]:::kspace
    C[️Phase Ramp\nMotion : A, f]:::corrupt
    D[Cartesian Undersampling\nFactor R]:::corrupt
    E[Complex Gaussian Noise\nSNR dB]:::corrupt
    F[Corrupted K-space\nreal + imaginary channels]:::kspace
    G[2D U-Net\nInput/Output : 2, H, W]:::model
    H[Reconstructed Image]:::output

    A -->|2D FFT| B
    B --> C
    B --> D
    B --> E
    C --> F
    D --> F
    E --> F
    F --> G
    G -->|iFFT| H

    classDef input    fill:#B497E7,stroke:#2C5F8A,color:#fff,rx:8
    classDef kspace   fill:#F6CF71,stroke:#4B3FA0,color:#fff,rx:8
    classDef corrupt  fill:#9EB9F3,stroke:#B07820,color:#fff,rx:8
    classDef model    fill:#F89C74,stroke:#2E8B57,color:#fff,rx:8
    classDef output   fill:#FF6B6B,stroke:#CC3333,color:#fff,rx:8
```
---

## Dataset

This project uses the **Whole-Spine Anatomical MRI dataset** (ds005616), available on:

- OpenNeuro: [https://openneuro.org/datasets/ds005616/versions/1.1.2](https://openneuro.org/datasets/ds005616/versions/1.1.2)
- GitHub: https://github.com/OpenNeuroDatasets/ds005616.git

For this project we use: 
- **Modality**: T2-weighted sagittal whole-spine MRI
- **Subjects**: 56
- **Resolution**: 1 mm³
- **Format**: NIfTI (.nii.gz), BIDS-compliant

<p align="center">
  <img src="README_figures/data_exemple.png" width="100">
</p>

Data access via Datalad:
```bash
datalad install https://github.com/OpenNeuroDatasets/ds005616.git
datalad get sub-*/anat/sub-*_T2w.nii.gz
```
---

## K-space corruption simulation

During a gradient echo (GRE) MRI acquisition, k-space lines are acquired sequentially over time. Respiratory motion between acquisitions induces a **phase shift** in each k-space line, modeled as:

$$\tilde{K}(k_x, k_y) = K(k_x, k_y) \cdot e^{-j2\pi k_x \cdot d(k_y)}$$

where $d(k_y)$ is the respiratory displacement at the time of acquisition of line $k_y$.

Combined with **Cartesian undersampling** (acceleration factor R) and **Gaussian noise**, this produces realistic corrupted k-space data that impose several typical artifacts on the reconstruted image.

🚨 Add figures

⚠️ Note: More detailled information obout the simulation process are provided inside the Kspace_corruption_simulation.ipynb notebook.
See [`notebooks/Kspace_corruption_simulation_vf.ipynb`](notebooks/Kspace_corruption_simulation_vf.ipynb).

## Deep learning model

### Trainning dataset

A training dataset was generated my varying the corruption paramters: 

| Parameter | Range | Description |
|---|---|---|
| A | 2.0, 5.0, 8.0 px | Motion amplitude |
| f | 12, 15, 18 breaths/min | Respiratory rate |
| R | 2, 4, 6 | Undersampling factor |
| SNR | 15, 20, 25 dB | Signal-to-noise ratio |

Total combinations: **81 per slice** → 231,417 training samples across 56 subjects.

Choice of simulation parameters:

- **Motion amplitude A:** The motion amplitude A (in pixels, assuming 1 mm isotropic resolution) spans the physiological range of respiratory spinal cord displacement measured in vivo. Studies have shown that breathing induces spinal cord displacements of up to 10 mm in the anterior-posterior direction at 3T. At 1 mm/px resolution (ds005616), this translates to A = 1–10 px. We choose to use 2, 5 et 8 px displacement to cover a large range of displacement, 8px being already an important displacement, we choose to not done the 10px being rare et intense above all that the displacement here are applied to all the image. 

> *Verma, T. and Cohen-Adad, J. (2014). Effect of respiration on the B0 field in the human spinal cord at 3T. Magnetic Resonance in Medicine, 72: 1629–1636. https://doi.org/10.1002/mrm.25075*

- **Respiratory rate f:** The normal respiratory rate for adults at rest ranges from 12 to 20 breaths/min. We use f = 12, 15, 18 breaths/min to cover this physiological range.

> *Sapra A, Malik A, Bhandari P. Vital Sign Assessment. [Updated 2023 May 1]. In: StatPearls [Internet]. Treasure Island (FL): StatPearls Publishing; 2026 Jan-. Available from: https://www.ncbi.nlm.nih.gov/books/NBK553213/*

- **Undersampling factor R:** R = 2, 4, 6 corresponds to acquiring around 50%, 25% and 17% of k-space lines respectively, covering the typical acceleration factors used in clinical accelerated MRI protocols.

- **Signal-to-noise ratio SNR:** We simulate SNR values of 15, 20 and 25 dB. An SNR above 20 dB is generally considered sufficient for diagnostic image quality.

> *McRobbie, D.W., Moore, E.A., Graves, M.J., Prince, M.R. (2017). MRI from Picture to Proton. Cambridge University Press. — "As a rule-of-thumb an SNR higher than 20:1 offers little image quality advantage to the observer"*


### Model Architecture

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
- **Scheduler**: ReduceLROnPlateau (patience=5, factor=0.5) 🚨 CHECK TO DO
- **Input normalization**: divided by max(|clean k-space|)

### Model trainning

🚨 Add info (time, batch, gpu, server)

---
## Results 

### Model evaluation

🚨 Add figures

### Model inference

🚨 Add figures

### Limitations
- Motion model is **1D rigid translation** only (no rotation, no non-rigid deformation)


---
## Repository Structure

```
Sarrazin_project/
│
├── ds005616/                          # Dataset
│
├── src/
│   ├── Kspace_simulation.py           # K-space corruption pipeline
│   ├── Utils.py                       # Metrics and utilities
│   ├── Unet_model.py                  # 2D U-Net architecture
│   └── Unet_train.py                  # Training loop
│
├── notebooks/
│   ├── Kspace_corruption_simulan_vf.ipynb   # Step-by-step simulation
│   ├── Unet_inference.ipynb                    # Inference & visualization
│   └── Unet_analysis.ipynb                   Training curves & metrics
	Gif.ipynb 				# Breathing motion simulation gif
│
├── training_data/
│   ├── mfest_v2.csv                # Dataset manifest (path, TR, TE, H, W, params)
│   └── splits.json                    # Train/val/test subject splits (70/15/15)
│
├── results/
│  ├── unet_best.pt               # Best moint
│  ├── training_history.csv       # Metrics per epoch
│  └─figures_example/           # Generated figures
│
├── train_full.sh                      # SLURM job script (Alliance Canada)
└── README.md
```
Additional information:

- manifest.csv: one row per (subject × slice × corruption combo) with image path, acquisition parameters (TR, TE), original dimensions (H, W), and corruption parameters (A, f, R, SNR).
- splits.json: Reproducible subject split: 70% train / 15% val / 15% test fixed at random seed 42 -> ensures no data leakage between sets.

---
## Reproductibility 

### Installation & Training

The following set-up use the server Narval on Alliance Canada, you can change it to the server name who want to use

### 1. Clone the repository

```bash
git clone https://github.com/annaellesarrazin/Sarrazin_project.git
cd Sarrazin_project
```

### 2. Set up the environment

```bash
conda create -n brainhack python=3.10
conda activate brainhack
pip install -r requirements.txt
```

### 3. Download the dataset

```bash
datalad install https://github.com/OpenNeuroDatasets/ds005616.git
cd ds005616
datalad get sub-*/anat/sub-*_T2w.nii.gz
cd ..
```

### 4. Transfer data to Alliance Canada (Narval)

```bash
rsync -av --progress \
    /path/to/Sarrazin_project/ \
    username@narval.alliancecan.ca:/scratch/username/Sarrazin_project/
```

### 5. Set up the environment on Narval

```bash
# Connect to Narval
ssh username@narval.alliancecan.ca

# Load Python module
module load python/3.10

# Create virtual environment
virtualenv ~/brainhack/brainhack
source ~/brainhack/brainhack/bin/activate

# Install dependencies
pip install torch torchvision
pip install -r requirements.txt
```

### 5. Train on Narval

```bash
# Connect to Narval
ssh username@narval.alliancecan.ca

# Go to the folder Sarrazin_project
cd /scratch/username/Sarrazin_project

# Submit the job — environment activation is handled by train_full.sh
sbatch train_full.sh

# Monitor training
squeue -u $USER          # check job status
tail -f logs/unet_*.out  # follow live logs
```

### 6. Retrieve results

```bash
# From your local machine
scp username@narval.alliancecan.ca:/scratch/username/Sarrazin_project/results/full_run/{unet_best.pt,training_history.csv} \
    results/
```

### 7. Inference & visualization

Open `notebooks/Unet_inference.ipynb` and set:

```python
MODEL_PATH  = Path('results/unet_best.pt')
SPLITS_PATH = Path('training_data/splits.json')
```

### 8. Analyze training

Open `notebooks/Unet_analysis.ipynb` and set:

```python
HISTORY_PATH = Path('results/training_history.csv')
```

---
## References


---
## Author

**Annaelle Sarrazin** — Brainhack School 2026
