# ======================
# exampler commands on miniImageNet
# ======================

# supervised pre-training
python train_supervised.py --trial pretrain --model_path /path/to/save --tb_path /path/to/tensorboard --data_root /path/to/data_root

python3 train_supervised.py --trial pretrain --num_workers 1 --data_root ../deep-conus-master/data/ --dataset miniDeepConus

# distillation
# setting '-a 1.0' should give similar performance
python train_distillation.py -r 0.5 -a 0.5 --path_t /path/to/teacher.pth --trial born1 --model_path /path/to/save --tb_path /path/to/tensorboard --data_root /path/to/data_root

python3 train_distillation.py -r 0.5 -a 0.5 --path_t ./models_pretrained/resnet12_miniDeepConus_lr_0.05_decay_0.0005_trans_A_trial_pretrain/resnet12_last.pth --trial born1 --num_workers 1 --data_root ../deep-conus-master/data/ --dataset miniDeepConus --epochs 10

# evaluation
python eval_fewshot.py --model_path /path/to/student.pth --data_root /path/to/data_root

python3 eval_fewshot.py --model_path ./models_distilled/S-resnet12_T-resnet12_miniDeepConus_kd_r-0.5_a-0.5_b-0_trans_A_born1/resnet12_last.pth --num_workers 1 --data_root ../deep-conus-master/data/ --dataset miniDeepConus

python3 eval_fewshot.py --model_path ./models_pretrained/resnet12_miniDeepConus_lr_0.05_decay_0.0005_trans_A_trial_pretrain/resnet12_last.pth --num_workers 1 --data_root ../deep-conus-master/data/ --dataset miniDeepConus