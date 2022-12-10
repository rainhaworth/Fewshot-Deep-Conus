# Fewshot-Deep-Conus
Recent few-shot learning methods adapted for atmospheric spatial data contained in the Deep Conus dataset. Repositories used:

https://github.com/mariajmolina/deep-conus

https://github.com/yinboc/few-shot-meta-baseline

https://github.com/WangYueFt/rfs/

https://github.com/bertinetto/r2d2

# Generating data

1. Download data from Google Drive and move or extract to `/deep-conus/data/`.

2. Navigate to `/deep-conus/` then run `decompose.py`. This will decompose the larger files into individual `.pickle` files, each of which contains data for one sample. (This may take several hours, but does not need to be repeated.)

3. Run `splitmulti.py`. Adjust parameters as desired. Use `--help` for a list of parameters. This will produce 3 `.pickle` files containing dictionaries for the training, validation, and testing splits. Note that in order to maintain compatibility with the few-shot methods, each split **must have at least 5 classes each** and each class **must contain at least 20 samples.**

4. Update the timestamp. Currently, this is hardcoded and should be changed in `/few-shot-meta-baseline/datasets/mini_deep_conus.py`, `/rfs/dataset/mini_deep_conus.py`, and `r2d2/fewshots/data/mini_deep_conus.py` before running experiments.

# Training and Evaluation

## Few-shot-meta-baseline

Navigate to `/few-shot-meta-baseline/`. Always run Classifier-Baseline before Meta-Baseline. Edit `/configs/test_few_shot.yaml` if you wish to evaluate with Classifier-Baseline instead of Meta-Baseline.

Train Classifier-Baseline: `python train_classifier.py --config configs/train_classifier_mini_dc.yaml`

Train Meta-Baseline: `python train_meta.py --config configs/train_meta_mini_dc.yaml`

Evaluate 1-shot: `python test_few_shot.py --shot 1`

Evaluate 5-shot: `python test_few_shot.py --shot 5`

## Representations for Few-Shot Learning (RFS)

Navigate to `/rfs/`. If you have enough memory or are using a cluster, set `--num_workers 8`.

Train: `python train_supervised.py --trial pretrain --num_workers 1 --data_root ../deep-conus/data/ --dataset miniDeepConus`

Self-distillation: `python train_distillation.py -r 0.5 -a 0.5 --path_t ./models_pretrained/resnet12_miniDeepConus_lr_0.05_decay_0.0005_trans_A_trial_pretrain/resnet12_last.pth --trial born1 --num_workers 1 --data_root ../deep-conus/data/ --dataset miniDeepConus`

Evaluate 1-shot: `python eval_fewshot.py --model_path ./models_distilled/S-resnet12_T-resnet12_miniDeepConus_kd_r-0.5_a-0.5_b-0_trans_A_born1/resnet12_last.pth --num_workers 1 --data_root ../deep-conus/data/ --dataset miniDeepConus`

Evaluate 5-shot: `python eval_fewshot.py --model_path ./models_distilled/S-resnet12_T-resnet12_miniDeepConus_kd_r-0.5_a-0.5_b-0_trans_A_born1/resnet12_last.pth --num_workers 1 --data_root ../deep-conus/data/ --dataset miniDeepConus --n_shots 5`

## Meta-learning with differentiable closed-form solvers

Navigate to `/r2d2/scripts/train/`. I have included the set of parameters I have found to be most effective on Deep Conus, but feel free to experiment with other options. Note that this method requires that you have enough RAM available to store your entire training set and validation set.

Train: `python run_train.py --log.exp_dir mini_r2d2 --data.dataset minideepconus --data.way 5 --data.root_dir ../../../deep-conus/data --model.drop 0.1 --base_learner.learn_lambda True --base_learner.lambda_base 2 --base_learner.init_lambda 8 --base_learner.adj_base 2`

Evaluate 1-shot: `python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 1 --model.model_path ../train/results/mini_r2d2/best_model.1shot.t7`

Evaluate 5-shot: `python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 5 --model.model_path ../train/results/mini_r2d2/best_model.5shot.t7`
