# ECCV2022-OOD-CV-Challenge-Classification-Track-USTC-IAT-United
Competition Open Source Solutions


## 1. Environment setting 

### 1.0. Package
* Several important packages
    - torch == 1.8.1+cu111
    - trochvision == 0.9.1+cu111

### 1.1. Dataset
In the classification track, we use only the OOD classification and detection data and labels:
* [ECCV-OOD](https://github.com/eccv22-ood-workshop/ROBIN-dataset)

### 1.2. OS
- [x] Windows10
- [x] Ubuntu20.04
- [x] macOS (CPU only)

## 2. Train
- [x] Single GPU Training
- [x] DataParallel (single machine multi-gpus)
- [ ] DistributedDataParallel

(more information: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

### 2.1. data
train data and test data structure:  
```
├── data/
│   ├── ROBINv1.1-cls-pose
│   ├── ROBINv1.1-det
│   ├── phase2-cls
│   ├── phase2-det
│   └── pseudo
```
  
Training sets and test sets are distributed with CSV labels corresponding to them.

### 2.2. run.
Divided into three stages of training, the code files used are src/train.sh, src/test.sh and src/mix_final.ipynb

1. 1st stage : train 3 models and ensembel them to output 1st round Pseudo-labeling
```
python -m torch.distributed.launch --master_port 145640 --nproc_per_node=4 train.py -c configs/deit_large_384.yaml \
        --output ../../output/deit/large_384/v7 \
        --batchformer


python -m torch.distributed.launch --master_port 145647 --nproc_per_node=2 train.py -c configs/convnext_large.yaml \
        --output ../../output/convnext/large_224/v5 \
        --model-ema

python -m torch.distributed.launch --master_port 145640 --nproc_per_node=4 train.py -c configs/volo_d5_512.yaml \
        --output ../../output/volo/d5_512/v3

python test_final.py --model deit3_large_patch16_384 \
              --num-gpu 4 \
              --img-size 384 \
              --num-classes 10 \
              --batch-size 200 \
              --checkpoint ../../output/deit/large_384/v7/20220913-101337-deit3_large_patch16_384-384/checkpoint-37.pth.tar \
              --output_path ../final_result/deit_large_384_bf/ \
              --scoreoutput_path ../final_result/deit_large_384_bf/

python test_final.py --model volo_d5_512 \
              --num-gpu 4 \
              --img-size 512 \
              --num-classes 10 \
              --batch-size 80 \
              --checkpoint ../../output/volo/d5_512/v3/20220910-202730-volo_d5_512-512/checkpoint-12.pth.tar \
              --output_path ../final_result/volo_d5_512/ \
              --scoreoutput_path ../final_result/volo_d5_512/

python test_final.py --model convnext_large \
              --num-gpu 2 \
              --img-size 224 \
              --num-classes 10 \
              --batch-size 200 \
              --checkpoint ../../output/convnext/large_224/v5/20220902-151019-convnext_large-224/checkpoint-45.pth.tar \
              --output_path ../final_result/convnext_large_224/ \
              --scoreoutput_path ../final_result/convnext_large_224/

run src/mix_final.ipynb
```

2. 2nd stage : train 2 models and ensembel them to output 2nd round Pseudo-labeling
```
python -m torch.distributed.launch --master_port 145640 --nproc_per_node=4 train_final.py -c configs/deit_large_384.yaml \
        --output ../../output/deit/large_384/v11 \
        --batchformer
 
python -m torch.distributed.launch --master_port 145647 --nproc_per_node=2 train_final.py -c configs/convnext_large.yaml \
        --output ../../output/convnext/large_224/v14 \
        --model-ema
 
python test_final.py --model deit3_large_patch16_384 \
              --num-gpu 4 \
              --img-size 384 \
              --num-classes 10 \
              --batch-size 200 \
              --checkpoint ../../output/deit/large_384/v11/20220928-204730-deit3_large_patch16_384-384/checkpoint-13.pth.tar \
              --output_path ../final_result/deit_large_384_bf_pseduo_5_epoch13/ \
              --scoreoutput_path ../final_result/deit_large_384_bf_pseduo_5_epoch13/

python test_final.py --model convnext_large \
              --num-gpu 2 \
              --img-size 224 \
              --num-classes 10 \
              --batch-size 200 \
              --checkpoint ../../output/convnext/large_224/v14/20220927-165319-convnext_large-224/checkpoint-71.pth.tar \
              --output_path ../final_result/convnext_large_pseduo_5/ \
              --scoreoutput_path ../final_result/convnext_large_pseduo_5/
              
run src/mix_final.ipynb
```


3. 3rd stage : train 1 model 
```
python -m torch.distributed.launch --master_port 145640 --nproc_per_node=4 train_final2.py -c configs/volo_d5_512.yaml \
        --output ../../output/volo/d5_512/v5

python test_final.py --model volo_d5_512 \
              --num-gpu 4 \
              --img-size 512 \
              --num-classes 10 \
              --batch-size 80 \
              --checkpoint ../../output/volo/d5_512/v5/20220930-080545-volo_d5_512-512/checkpoint-6.pth.tar \
              --output_path ../final_result/volo_d5_512_second_8/ \
              --scoreoutput_path ../final_result/volo_d5_512_second_8/
```



## 3. Evaluation
for details, see src/mix_final.ipynb
```
run src/mix_final.ipynb
```


## 4. Challenge's final checkpoints
It can be downloaded from Google Cloud Disk: https://drive.google.com/file/d/1nx5pE1Axj-tzKkAWtVt9HQu7WF1PbjfF/view?usp=sharing

It can be directly used for model ensemble reasoning and get final result.

### Acknowledgment

* Thanks to [timm](https://github.com/rwightman/pytorch-image-models) for Pytorch implementation.
