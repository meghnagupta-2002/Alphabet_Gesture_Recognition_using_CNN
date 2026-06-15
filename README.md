# Alphabet Gesture Recognition using CNN

Real-time **Sign Language to Text** conversion using a Convolutional Neural Network. The
system captures hand gestures from a webcam, classifies them into the 26 letters of the
alphabet (A–Z) plus a *blank* class, and assembles the recognized letters into words and
sentences on screen.

The project covers the full pipeline:

1. **Collect** your own gesture dataset from a webcam (`collectdata.py`)
2. **Train** a CNN on the collected images (`train.py`)
3. **Run** a real-time GUI that recognizes gestures and builds text (`app.py`)

---

## Demo / How it works

A green ROI (Region of Interest) box is drawn on the webcam feed. The hand inside the box is
converted to a clean black-and-white silhouette using a sequence of image filters:

- Convert to grayscale
- Gaussian blur to reduce noise
- Adaptive Gaussian thresholding
- Otsu binary thresholding

The resulting binary image is what the CNN sees, both during data collection and at inference
time, which keeps the model robust to lighting and background changes.

At inference, predictions are stabilized over consecutive frames before a letter is committed
to the current word. A *blank* gesture acts as a space, separating words in the output
sentence.

---

## Repository structure

| File | Description |
|------|-------------|
| `collectdata.py` | Webcam tool to capture and save labeled gesture images into a dataset folder structure. |
| `train.py` | Builds, trains, and saves the CNN model. |
| `app.py` | Tkinter GUI for real-time gesture recognition and sentence building. |
| `README.md` | This file. |

---

## Requirements

- Python 3.x
- A working webcam

Install the dependencies:

```bash
pip install tensorflow keras opencv-python numpy pillow matplotlib
```

> `tkinter` ships with most standard Python installations. On some Linux distributions you may
> need to install it separately (e.g. `sudo apt install python3-tk`).

---

## 1. Collecting data

Run the collection tool:

```bash
python collectdata.py
```

This creates the following directory structure automatically:

```
Dataset/
├── Training Data/
│   ├── 0/   (blank)
│   ├── A/
│   ├── B/
│   └── ... Z/
└── Test Data/
    ├── 0/
    ├── A/
    └── ... Z/
```

While the window is open:

- Position your hand inside the green box.
- Press a letter key (`a`–`z`) to save the current processed frame into that letter's folder.
- Press `0` to save a *blank* (no-gesture) sample.
- Press `Esc` to quit.

The on-screen counters show how many samples you have collected for each class. To build the
**test set**, change this line in `collectdata.py`:

```python
DataFor = 'Training Data'   # change to 'Test Data'
```

Collect a balanced number of images per class for best results.

---

## 2. Training the model

Once you have a dataset, train the CNN:

```bash
python train.py
```

### Model architecture

The network expects **128×128 grayscale** images and outputs **27 classes** (26 letters + blank):

```
Input (128 × 128 × 1)
 → Conv2D(32, 3×3, ReLU) → MaxPool(2×2)
 → Conv2D(32, 3×3, ReLU) → MaxPool(2×2)
 → Flatten
 → Dense(128, ReLU) → Dropout(0.40)
 → Dense(96,  ReLU) → Dropout(0.40)
 → Dense(64,  ReLU)
 → Dense(27,  Softmax)
```

- **Optimizer:** Adam
- **Loss:** Categorical cross-entropy
- **Data augmentation:** rescaling, shear, zoom, horizontal flip (via `ImageDataGenerator`)

After training, the model is saved as:

- `model-bw.json` — network architecture
- `model-bw.h5` — trained weights

> **Note:** `steps_per_epoch` and `validation_steps` in `train.py` are set to the number of
> images in the training/test sets. Update these to match the size of your own dataset.

---

## 3. Running the application

```bash
python app.py
```

The GUI (`app.py`) loads the trained model and displays:

- The live webcam feed with the ROI box
- The processed binary image fed to the model
- The current predicted **character**
- The **word** built from committed characters

### Multi-model disambiguation

`app.py` loads a **main model** plus three specialized models that resolve commonly confused
gesture groups:

| Model file | Disambiguates |
|------------|---------------|
| `model-bw.{json,h5}`      | All 26 letters + blank (primary classifier) |
| `model-bw_dru.{json,h5}`  | D, R, U |
| `model-bw_tkdi.{json,h5}` | T, K, D, I |
| `model-bw_smn.{json,h5}`  | S, M, N |

When the primary model predicts a letter belonging to one of these ambiguous groups, the
corresponding specialized model casts the deciding vote. These model files are expected to
live in a `model/` directory (the app changes into it on startup).

---

## Notes & limitations

- The repository contains the **source code only** — datasets and pre-trained model files
  (`.h5` / `.json`) are not included. You must collect data and train the models, and place
  the trained model files in a `model/` directory before running `app.py`.
- Recognition quality depends heavily on consistent lighting and a clean background behind the
  hand, since the model is trained on thresholded silhouettes.
- A commented-out [hunspell](https://hunspell.github.io/) integration in `app.py` hints at an
  intended word-suggestion / autocorrect feature.

---

## Authors

Developed as a Computer Science & Engineering project by:

- Meghna Gupta
- Nitika Shrivastava
- Kaushik Pranay
- Dheeraj Sadani

**Project Guide:** Prof. Sanjay Pal · **Project Coordinator:** Prof. Imran Ali Khan ·
**HOD CSE:** Dr. Deepshikha Patel
