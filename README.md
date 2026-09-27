# 🎨 Anime AI vs Human-Made Illustration Detector

A research-oriented image classification project for distinguishing
**AI-generated anime illustrations** from **human-made illustrations**.

The repository contains the complete pipeline for dataset preparation,
preprocessing, model training, evaluation, statistical analysis, and a
Streamlit inference application.

## 🌐 Live Demo

Try the deployed detector here:

**https://anime-ai-detector.streamlit.app/**

The deployed application runs inference on CPU and provides three
analysis modes:

-   **MobileNetV3-Large --- Level 4 (Haar DWT)**
-   **DeiT-Tiny --- Level 3 (Discriminative Learning Rate)**
-   **Both models** for side-by-side comparison

> The deployed application is an inference interface. You do **not**
> need to download or prepare the training dataset just to use the web
> app.

------------------------------------------------------------------------

## 🧠 Models

The research focuses on two image-classification architectures:

  -----------------------------------------------------------------------
  Model                   Architecture            Main role in the
                                                  project
  ----------------------- ----------------------- -----------------------
  MobileNetV3-Large       Lightweight CNN         Efficient image
                                                  classification;
                                                  deployed with Level 4

  DeiT-Tiny               Vision Transformer /    Alternative
                          Transformer-based image architecture; deployed
                          classifier              with Level 3
  -----------------------------------------------------------------------

Both models are configured as a **binary classifier**:

-   `0 = AI-generated`
-   `1 = Human-made`

The project uses pretrained model weights and replaces the original
classification head with a two-class output.

------------------------------------------------------------------------

# 🚀 Quick Start

If you only want to run the detector locally, you do **not** need to
reproduce the complete research pipeline.

## 1. Clone the repository

``` bash
git clone https://github.com/mna1505/anime-ai-detector.git
cd anime-ai-detector
```

## 2. Create a virtual environment

### Windows

``` bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

The `.venv/` directory is a local Python environment and should not be
committed to GitHub.

## 3. Install the dependencies

``` bash
pip install -r requirements.txt
```

The current `requirements.txt` is intended for the Streamlit inference
application and contains:

``` text
streamlit
torch
torchvision
timm
Pillow
```

## 4. Run the Streamlit application

From the repository root:

``` bash
streamlit run src/app.py
```

Then open the local URL shown by Streamlit, normally:

``` text
http://localhost:8501
```

## 5. Use the detector

1.  Upload an anime illustration.
2.  Select an analysis mode:
    -   MobileNetV3-Large --- Level 4
    -   DeiT-Tiny --- Level 3
    -   Both models
3.  Click **Deteksi**.
4.  View:
    -   predicted class,
    -   AI probability,
    -   Human-made probability,
    -   inference latency.

The application loads the trained checkpoints from `models_modified/`.

------------------------------------------------------------------------

# 📁 Repository Structure

``` text
anime-ai-detector/
│
├── dataset/
│   ├── raw/
│   ├── prepared/
│   └── splits/
│
├── models/
│   └── ...                       # Baseline checkpoints
│
├── models_modified/
│   ├── level1/
│   ├── level2/
│   ├── level3/
│   │   └── deit_tiny/
│   │       └── best_model.pth
│   └── level4/
│       └── mobilenetv3/
│           └── best_model.pth
│
├── results/
│   └── ...                       # Baseline experiment results
│
├── results_modified/
│   ├── level1/
│   ├── level2/
│   ├── level3/
│   └── level4/
│
├── src/
│   ├── 01_prepare_dataset.py
│   ├── 02_preprocessing.py
│   ├── 02_preprocessing_modified.py
│   ├── 03_dataset.py
│   ├── 03_dataset_modified.py
│   ├── 04_train.py
│   ├── 04_train_modified.py
│   ├── 05_evaluate.py
│   ├── 05_evaluate_modified.py
│   ├── 06_gui_predict_dual_selected.py
│   ├── 07_statistical_visualization.py
│   ├── app.py
│   └── model_factory_modified.py
│
├── statistical_visualization/
│   └── ...                       # Generated research visualizations
│
├── requirements.txt
├── .gitignore
└── README.md
```

Local development folders such as `.venv/`, `.vscode/`, and
`__pycache__/` are intentionally excluded from the repository.

------------------------------------------------------------------------

# 🔬 Research Pipeline

The complete research workflow is more extensive than simply running the
Streamlit application.

The general pipeline is:

``` text
Raw Dataset
    ↓
Dataset Preparation
    ↓
Train / Validation / Test Split
    ↓
Baseline Training
    ↓
Level 1
    ↓
Level 2
    ↓
Level 3
    ↓
Level 4
    ↓
Evaluation
    ↓
Statistical Visualization
```

## Step 1 --- Prepare the raw dataset

The dataset preparation script is:

``` text
src/01_prepare_dataset.py
```

The script expects raw image directories under:

``` text
dataset/raw/
├── animedl2m_real/
│   └── images/
├── animedl2m_fake/
│   └── images/
├── novelai3/
│   └── images/
├── Pixiv_Popular/
│   └── images/
└── danbooru2021_SQLite/
    └── images/
```

The script currently contains a manual switch:

``` python
MODE = "PIXIV"
# MODE = "DANBOORU"
```

Therefore, choose the intended cross-generator human dataset before
running the preparation stage.

The preparation script performs filtering, fixed-size sampling, class
organization, and creation of the training/validation/test directory
structure.

The configured in-domain split is:

-   **2,250 images/class** for training
-   **500 images/class** for validation
-   **500 images/class** for in-domain testing

The cross-generator evaluation uses unseen human and AI sources prepared
separately from the in-domain training data.

Run:

``` bash
python src/01_prepare_dataset.py
```

------------------------------------------------------------------------

# 🧪 Step 2 --- Baseline Training

Baseline training is implemented in:

``` text
src/04_train.py
```

Run:

``` bash
python src/04_train.py
```

### Baseline configuration

The baseline uses:

-   pretrained MobileNetV3-Large and DeiT-Tiny,
-   binary classification,
-   standard image preprocessing,
-   Adam optimizer,
-   learning rate `1e-4`,
-   maximum `20` epochs,
-   early stopping with patience `5`,
-   batch size `32`,
-   random seed `42`.

### Baseline preprocessing

Images are processed using:

1.  Resize longest side
2.  Pad to square
3.  Resize to `224 × 224`
4.  Training augmentation:
    -   random horizontal flip,
    -   random rotation,
    -   color jitter
5.  ImageNet normalization

The baseline acts as the reference point for comparing the modified
training levels.

------------------------------------------------------------------------

# 📈 Step 3 --- Modified Training Levels

The modified training pipeline is implemented in:

``` text
src/04_train_modified.py
```

Both models can be trained for a specific level.

### MobileNetV3-Large

``` bash
python src/04_train_modified.py --level 1 --model mobilenetv3
python src/04_train_modified.py --level 2 --model mobilenetv3
python src/04_train_modified.py --level 3 --model mobilenetv3
python src/04_train_modified.py --level 4 --model mobilenetv3
```

### DeiT-Tiny

``` bash
python src/04_train_modified.py --level 1 --model deit_tiny
python src/04_train_modified.py --level 2 --model deit_tiny
python src/04_train_modified.py --level 3 --model deit_tiny
python src/04_train_modified.py --level 4 --model deit_tiny
```

Or train both models for one level:

``` bash
python src/04_train_modified.py --level 1 --all-models
python src/04_train_modified.py --level 2 --all-models
python src/04_train_modified.py --level 3 --all-models
python src/04_train_modified.py --level 4 --all-models
```

------------------------------------------------------------------------

# 🧩 What Changes from Baseline to Level 4?

The experiment is cumulative: each level introduces a new training or
preprocessing strategy.

  -----------------------------------------------------------------------
  Stage                               Main configuration
  ----------------------------------- -----------------------------------
  **Baseline**                        Pretrained model + standard
                                      preprocessing/augmentation + Adam

  **Level 1**                         AdamW + cosine annealing + stronger
                                      robustness augmentation

  **Level 2**                         Level 1 + MixUp + label smoothing +
                                      warm-up

  **Level 3**                         Level 2 + discriminative learning
                                      rate

  **Level 4**                         Level 3 optimizer policy + Haar DWT
                                      frequency-domain preprocessing
  -----------------------------------------------------------------------

## Baseline

The baseline provides the reference performance.

It uses a conventional RGB input pipeline and standard fine-tuning
setup. No MixUp, label smoothing, discriminative learning rate, or Haar
DWT is used.

## Level 1 --- Robust Training

Level 1 changes the optimization strategy to:

-   AdamW
-   cosine annealing learning-rate schedule

It also uses stronger image augmentation, including:

-   random resized crop,
-   horizontal flip,
-   rotation,
-   color jitter,
-   Gaussian blur,
-   JPEG compression,
-   random erasing.

The goal of this stage is to expose the model to more visual variations
during training.

## Level 2 --- MixUp + Label Smoothing + Warm-up

Level 2 builds on Level 1 and introduces:

-   **MixUp** with `alpha = 0.2`
-   **Label smoothing** with `0.10`
-   **Learning-rate warm-up** for the first `3` epochs

MixUp creates training examples by combining images and their labels,
while label smoothing prevents the classifier from relying on
excessively confident target probabilities.

## Level 3 --- Discriminative Learning Rate

Level 3 keeps the Level 2 strategy and introduces different learning
rates for different parts of the network:

-   backbone learning rate: `1e-5`
-   classification head learning rate: `1e-4`

This allows the pretrained backbone to be updated more conservatively
while the classification head learns more quickly.

## Level 4 --- Haar DWT

Level 4 keeps the optimizer policy from Level 3 and changes the input
representation.

A **one-level Haar Discrete Wavelet Transform (DWT)** is used to extract
high-frequency information from the image.

The final input becomes:

``` text
RGB + DWT high-frequency channel
= 4 input channels
```

The high-frequency representation is derived from the grayscale image
using the Haar wavelet components and normalized before being
concatenated with the RGB channels.

The model input layer is therefore modified from 3 channels to 4
channels for Level 4.

> Level 4 in this project does **not** use SAM. The Level 4 modification
> is the Haar DWT preprocessing while retaining the Level 3 optimizer
> policy.

------------------------------------------------------------------------

# 📊 Cross-Generator Accuracy

The following visualization compares classification accuracy from the
baseline through Level 4 under the **Cross Generator** condition.

![Cross-Generator Accuracy --- Baseline to Level
4](statistical_visualization/09_line_dot_statistics/classification/cross_generator_accuracy_line_dot.png)

### Recorded values in the visualization

  Stage        MobileNetV3-Large   DeiT-Tiny
  ---------- ------------------- -----------
  Baseline                56.80%      66.80%
  Level 1                 62.20%      64.40%
  Level 2                 62.10%      58.80%
  Level 3                 75.50%      71.80%
  Level 4                 77.50%      68.20%

The plot shows how the two architectures behave across the successive
experimental stages.

For **MobileNetV3-Large**, the plotted cross-generator accuracy
progresses from `56.80%` at baseline to `77.50%` at Level 4.

For **DeiT-Tiny**, the plotted accuracy is `66.80%` at baseline,
`64.40%` at Level 1, `58.80%` at Level 2, `71.80%` at Level 3, and
`68.20%` at Level 4.

These values are presented as the recorded experimental results in the
repository visualization; they should be interpreted in the context of
the specific cross-generator test set and experimental setup.

------------------------------------------------------------------------

# 🧪 Step 4 --- Evaluation

Baseline evaluation:

``` bash
python src/05_evaluate.py
```

Modified evaluation requires a level:

``` bash
python src/05_evaluate_modified.py --level 1 --all-models
python src/05_evaluate_modified.py --level 2 --all-models
python src/05_evaluate_modified.py --level 3 --all-models
python src/05_evaluate_modified.py --level 4 --all-models
```

A specific model can also be evaluated:

``` bash
python src/05_evaluate_modified.py --level 4 --model mobilenetv3
```

The evaluation pipeline records classification metrics and
inference-efficiency information for the configured test conditions.

------------------------------------------------------------------------

# 📊 Step 5 --- Statistical Visualization

The statistical visualization pipeline is implemented in:

``` text
src/07_statistical_visualization.py
```

Run:

``` bash
python src/07_statistical_visualization.py
```

The script combines baseline and modified-level results and generates
visualizations for:

-   classification performance,
-   confusion matrix,
-   changes relative to baseline,
-   cross-generator generalization,
-   inference/training efficiency,
-   memory usage,
-   computational metrics,
-   baseline-to-Level-4 comparisons.

The generated figures are stored under:

``` text
statistical_visualization/
```

------------------------------------------------------------------------

# 🖥️ Streamlit Application

The current web interface is:

``` text
src/app.py
```

The application is designed specifically for inference and uses CPU
execution.

### Available modes

#### 1. MobileNetV3-Large --- Level 4

Uses:

-   MobileNetV3-Large
-   Level 4 checkpoint
-   Haar DWT preprocessing
-   4-channel input: RGB + DWT

#### 2. DeiT-Tiny --- Level 3

Uses:

-   DeiT-Tiny
-   Level 3 checkpoint
-   discriminative learning rate configuration used during training
-   standard 3-channel RGB inference preprocessing for Level 3

#### 3. Both Models

Runs both configured checkpoints and displays their predictions for the
same uploaded image.

The application also reports inference latency for the selected
model(s).

------------------------------------------------------------------------

# 📦 Additional Dependencies for Full Research Reproduction

The current `requirements.txt` is optimized for the Streamlit inference
application.

If you want to reproduce the **training, evaluation, and
statistical-analysis pipeline**, the project scripts also use additional
scientific Python packages such as:

``` bash
pip install numpy pandas scikit-learn matplotlib psutil
```

For optional FLOPs/MACs complexity analysis:

``` bash
pip install fvcore thop
```

The legacy Gradio interface in `src/06_gui_predict_dual_selected.py`
additionally requires:

``` bash
pip install gradio
```

You do not need Gradio when using the current Streamlit application.

------------------------------------------------------------------------

# ⚠️ Important Notes

-   The training pipeline is substantially more resource-intensive than
    inference.
-   The Streamlit application is intended to make inference accessible
    without requiring users to train the models themselves.
-   The reported accuracy values are tied to the specific datasets,
    splits, preprocessing, checkpoints, and evaluation conditions used
    in this project.
-   An AI-vs-human classifier should be treated as a prediction system
    rather than absolute proof of an image's origin.
-   The repository contains both the original baseline pipeline and the
    modified Level 1--4 research pipeline.

------------------------------------------------------------------------

# 📚 Main Scripts

  -----------------------------------------------------------------------
  Script                              Purpose
  ----------------------------------- -----------------------------------
  `01_prepare_dataset.py`             Prepare raw datasets and create
                                      train/validation/test splits

  `02_preprocessing.py`               Baseline preprocessing

  `02_preprocessing_modified.py`      Level 1--4 preprocessing

  `03_dataset.py`                     Baseline DataLoader

  `03_dataset_modified.py`            Modified Level 1--4 DataLoader

  `04_train.py`                       Baseline training

  `04_train_modified.py`              Level 1--4 training

  `05_evaluate.py`                    Baseline evaluation

  `05_evaluate_modified.py`           Level 1--4 evaluation

  `06_gui_predict_dual_selected.py`   Legacy Gradio inference interface

  `07_statistical_visualization.py`   Statistical analysis and
                                      visualization

  `app.py`                            Current Streamlit inference
                                      application

  `model_factory_modified.py`         MobileNetV3-Large and DeiT-Tiny
                                      model factory
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 👤 Project

**Repository:**\
https://github.com/mna1505/anime-ai-detector

**Live Demo:**\
https://anime-ai-detector.streamlit.app/

------------------------------------------------------------------------

## License

See the repository contents and project documentation for the applicable
licensing information.
