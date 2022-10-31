# Fewshot-Deep-Conus
Recent few-shot learning methods adapted for atmospheric spatial data contained in the Deep Conus dataset. Repositories used:

https://github.com/mariajmolina/deep-conus

https://github.com/yinboc/few-shot-meta-baseline

https://github.com/WangYueFt/rfs/

https://github.com/bertinetto/r2d2

# Generating data

1. Download data from Google Drive and move/extract to `/deep-conus/data/`.

2. Open `deep-conus/fewshot.ipynb` as a Jupyter Notebook.

3. Navigate to the cell that defines split_data_val(). Make any necessary revisions, e.g. to the filename. Some sections of the code might be block commented out; if so, un-comment those sections.

4. Run split_data_val() with your desired parameters. This should produce 3 `.pickle` files contianing data and 3 `.txt` files containing labels. Note that in order to maintain compatibility with the few-shot methods, the train, test, and validation splits **must have at least 5 classes each** and each class **must contain at least 20 samples.**

5. Update the timestamp. Currently, this is hardcoded and should be changed in `/few-shot-meta-baseline/datasets/mini_deep_conus.py` and `/rfs/dataset/mini_deep_conus.py`.

# Running scripts

## Few-shot-meta-baseline

Navigate to `/few-shot-meta-baseline/`. Always run Classifier-Baseline before Meta-Baseline. Edit `/configs/test_few_shot.yaml` to evaluate with Classifier-Baseline instead of Meta-Baseline.

Train Classifier-Baseline: `python train_classifier.py --config configs/train_classifier_mini_dc.yaml`

Train Meta-Baseline: `python train_meta.py --config configs/train_meta_mini_dc.yaml`

Evaluate: `python test_few_shot.py --shot 1`; for n-shot learning, use `--shot n`

## Representations for Few-Shot Learning (RFS)

Navigate to `/rfs/`. If you have enough memory or are using a cluster, set `--num_workers 8`.

Train: `python train_supervised.py --trial pretrain --num_workers 1 --data_root ../deep-conus/data/ --dataset miniDeepConus`

Self-distillation: `python train_distillation.py -r 0.5 -a 0.5 --path_t ./models_pretrained/resnet12_miniDeepConus_lr_0.05_decay_0.0005_trans_A_trial_pretrain/resnet12_last.pth --trial born1 --num_workers 1 --data_root ../deep-conus/data/ --dataset miniDeepConus`

Evaluate: `python eval_fewshot.py --model_path ./models_distilled/S-resnet12_T-resnet12_miniDeepConus_kd_r-0.5_a-0.5_b-0_trans_A_born1/resnet12_last.pth --num_workers 1 --data_root ../deep-conus/data/ --dataset miniDeepConus`

## Meta-learning with differentiable closed-form solvers

Not implemented; codebase is older and requires more effort to adapt.

Once implemented, commands used will be similar to those found at `/r2d2/scripts/train/deep-conus.sh`.
