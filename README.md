# CNN -> ViT classifier for ASD facial-image screening

Image -> `SpatialCNNExtractor` (keeps the 2-D feature map) -> tokens -> ViT -> P(ASD).
CNN and ViT are trained together, from scratch, on your dataset. Research code, not a diagnostic tool.

| File | Role |
|---|---|
| `cnn_feature_extractor.py` | spatially-preserving CNN (ConvNeXt-style blocks, SE + spatial attention) |
| `vit_asd_classifier.py` | ViT head and `CNNVITASDClassifier` (index 0 = Non-Autistic, 1 = Autistic) |
| `train_asd.py` | training + evaluation (subject-aware split, AUROC / sensitivity / specificity) |
| `predict_asd.py` | score new images with a trained checkpoint |

Class folders are detected automatically (`Autistic`/`ASD`/`autism` vs `Non_Autistic`/`TD`/`control`).
Pre-split `train/valid/test` folders are pooled and re-split by the script.

## Kaggle

1. Add your image dataset to the notebook (**Add Input**). Turn on **Accelerator: GPU** and **Internet** (needed for `git clone`).
2. Cells:
```python
!git clone https://github.com/<your-username>/asd-cnn-vit.git
%cd asd-cnn-vit
!find /kaggle/input -maxdepth 3 -type d | head -30   # find the folder that directly holds the two class folders
```
3. Smoke test, then the real run (replace the path):
```python
!python train_asd.py --data_root /kaggle/input/<dataset>/<folder> --epochs 2 --out_dir /kaggle/working/runs
!python train_asd.py --data_root /kaggle/input/<dataset>/<folder> --out_dir /kaggle/working/runs
```
Outputs (`best.pt`, `last.pt`, `run_log.json`, prediction CSVs) land in `/kaggle/working/runs/<run>/`. Use
**Save Version** to keep them; there is no resume-from-checkpoint, so a session that hits the time limit restarts that run.

4. Predict on new images:
```python
!python predict_asd.py --checkpoint /kaggle/working/runs/<run>/best.pt --input /kaggle/input/<new_images> --out_csv /kaggle/working/preds.csv
```

No internet / private repo? Upload the repo as a zip via **Add Input -> Upload -> New Dataset**, then `%cd` into it under `/kaggle/input/`.

## Local
```
pip install -r requirements.txt
python train_asd.py --data_root /path/to/dataset_root
```
Run `python train_asd.py -h` for every hyper-parameter; recommended starting values are in the docstring.

## Read before trusting results
- Splits are by subject only if IDs can be read from filenames or sub-folders. Otherwise it falls back to an image-level split
  (it warns loudly) and every metric is an upper bound for unseen children.
- ~6M parameters trained from scratch on a few thousand images will overfit; compare against a pretrained backbone.
- Default hyper-parameters are conventional starting points, not tuned on ASD data.
- Tested on CPU with synthetic data. GPU mixed precision (fp16/bf16) and CUDA determinism are untested.
- Do not commit the dataset or any child images to this repo (`.gitignore` blocks common image types).
