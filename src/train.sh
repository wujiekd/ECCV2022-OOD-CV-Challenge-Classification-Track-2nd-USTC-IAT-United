# 1st stage : train 3 models 
#CUDA_VISIBLE_DEVICES=0,1,2,3 nohup sh train.sh > nohup_deit_large_384_v7.out &
python -m torch.distributed.launch --master_port 145640 --nproc_per_node=4 train.py -c configs/deit_large_384.yaml \
        --output ../../output/deit/large_384/v7 \
        --batchformer

# #CUDA_VISIBLE_DEVICES=0,1 nohup sh train.sh > nohup_convnext_large_224_v5.out &
# python -m torch.distributed.launch --master_port 145647 --nproc_per_node=2 train.py -c configs/convnext_large.yaml \
#         --output ../../output/convnext/large_224/v5 \
#         --model-ema

# #CUDA_VISIBLE_DEVICES=0,1,2,3 nohup sh train.sh > nohup_volo_d5_512_v3.out &
# python -m torch.distributed.launch --master_port 145640 --nproc_per_node=4 train.py -c configs/volo_d5_512.yaml \
#         --output ../../output/volo/d5_512/v3



# # 2nd stage : 1st round Pseudo-labeling  
# #CUDA_VISIBLE_DEVICES=0,1,2,3 nohup sh train.sh > nohup_deit_large_384_v11.out &
# python -m torch.distributed.launch --master_port 145640 --nproc_per_node=4 train_final.py -c configs/deit_large_384.yaml \
#         --output ../../output/deit/large_384/v11 \
#         --batchformer
 
# #CUDA_VISIBLE_DEVICES=0,1 nohup sh train.sh > nohup_convnext_large_224_v14.out &
# python -m torch.distributed.launch --master_port 145647 --nproc_per_node=2 train_final.py -c configs/convnext_large.yaml \
#         --output ../../output/convnext/large_224/v14 \
#         --model-ema


# # 3rd stage : 2st round Pseudo-labeling  
# #CUDA_VISIBLE_DEVICES=0,1,2,3 nohup sh train.sh > nohup_volo_d5_512_v5.out &
# python -m torch.distributed.launch --master_port 145640 --nproc_per_node=4 train_final2.py -c configs/volo_d5_512.yaml \
#         --output ../../output/volo/d5_512/v5
