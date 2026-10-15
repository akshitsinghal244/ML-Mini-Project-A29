# Bird Species Identification from an Image

UE24CS352A Machine Learning mini-project · Problem #89 · Team A-29 (Sec A)
Akshit Singhal (PES1UG24AM027) · Arya C A (PES1UG24AM053)

Identify the species (1 of 200) in a bird photo on **CUB-200-2011**.
A classical baseline (hand-made features) is compared with a ResNet-50 fine-tuned from ImageNet weights.

## Results (test set, 5,794 images, 200 classes)

| Method | Top-1 | Top-5 |
|---|---|---|
| Baseline: k-NN | 2.7% | 8.4% |
| Baseline: Linear SVM | 4.0% | 13.1% |
| Baseline: RBF-SVM | 5.5% | 16.2% |
| Baseline: Random Forest | 9.1% | 25.4% |
| CNN phase 1 (frozen backbone) | 62.0% | 88.2% |
| **CNN phase 2 (fine-tuned layer4)** | **78.3%** | **93.4%** |

Figures: `results/training_curves.png`, `results/confusion_matrix.png`.
Most confused: Common Raven -> American Crow, Fish Crow -> American Crow, California Gull -> Western Gull, Forsters Tern -> Common Tern.

## Method

- **Data:** 5,094 train / 900 validation (15% of the official train split, stratified) / 5,794 test (official test split). Full images, no bounding boxes.
- **Baseline:** HSV histogram + HOG + LBP (2,286-d) -> Linear SVM / RBF-SVM / Random Forest / k-NN.
- **CNN:** ResNet-50 (ImageNet) with a new `Linear(2048, 200)` head. Phase 1: backbone frozen, 5 epochs. Phase 2: `layer4` + head, 12 epochs.
  AdamW, cosine LR, label smoothing 0.1, mixed precision, best epoch chosen on validation.

## Repository layout

```
notebooks/bird_species_kaggle.ipynb   full pipeline: data, baseline, CNN, evaluation, demo
src/baseline.py                       standalone baseline script (original CUB file layout)
src/predict.py                        top-k prediction for one image using the trained weights
results/                              result figures (+ results.json)
requirements.txt
```

## How to run

**Kaggle (what we used)** - notebook: https://www.kaggle.com/code/akshitsinghal201/ml-mini-project-birds-a29/notebook?scriptVersionId=355058474

1. Create a Kaggle notebook from `notebooks/bird_species_kaggle.ipynb`.
2. Settings: Accelerator = GPU, Internet = On. Add a CUB-200-2011 dataset via *Add Data*.
3. Run all cells (~16 min). Outputs: `results.json`, figures, `bird_model.pth`.
   Quick test: set `NUM_CLASSES = 20`, `EPOCHS_P2 = 2` in the config cell.

**Local setup**

```
pip install -r requirements.txt
python src/baseline.py --root data/CUB_200_2011 --num-classes 20
python src/predict.py photo.jpg --weights bird_model.pth --topk 3
```

Dataset: https://data.caltech.edu/records/65de6-vp158 (extract to `data/CUB_200_2011/`).
Trained weights (`bird_model.pth`, ~94 MB) are not stored in git: [add download link].

## Limitations

Single run, no hyper-parameter search. CUB test images may also be in ImageNet, so the pretrained backbone may have seen some.

## References

- Wah et al., *The Caltech-UCSD Birds-200-2011 Dataset*, 2011.
- He et al., *Deep Residual Learning for Image Recognition*, 2015.
- Stanford CS229 (2014) project "Bird Species Identification from an Image" (problem source).
