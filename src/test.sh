# # 1st stage : infer 3 models
# # CUDA_VISIBLE_DEVICES=0,1,2,3 sh test.sh

# python test_final.py --model deit3_large_patch16_384 \
#               --num-gpu 4 \
#               --img-size 384 \
#               --num-classes 10 \
#               --batch-size 200 \
#               --checkpoint ../../output/deit/large_384/v7/20220913-101337-deit3_large_patch16_384-384/checkpoint-37.pth.tar \
#               --output_path ../final_result/deit_large_384_bf/ \
#               --scoreoutput_path ../final_result/deit_large_384_bf/

# python test_final.py --model volo_d5_512 \
#               --num-gpu 4 \
#               --img-size 512 \
#               --num-classes 10 \
#               --batch-size 80 \
#               --checkpoint ../../output/volo/d5_512/v3/20220910-202730-volo_d5_512-512/checkpoint-12.pth.tar \
#               --output_path ../final_result/volo_d5_512/ \
#               --scoreoutput_path ../final_result/volo_d5_512/

# python test_final.py --model convnext_large \
#               --num-gpu 2 \
#               --img-size 224 \
#               --num-classes 10 \
#               --batch-size 200 \
#               --checkpoint ../../output/convnext/large_224/v5/20220902-151019-convnext_large-224/checkpoint-45.pth.tar \
#               --output_path ../final_result/convnext_large_224/ \
#               --scoreoutput_path ../final_result/convnext_large_224/

# 2nd stage : infer 2 models
# CUDA_VISIBLE_DEVICES=0,1,2,3 sh test.sh

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

# 3rd stage : infer 1 model
# CUDA_VISIBLE_DEVICES=0,1,2,3 sh test.sh
python test_final.py --model volo_d5_512 \
              --num-gpu 4 \
              --img-size 512 \
              --num-classes 10 \
              --batch-size 80 \
              --checkpoint ../../output/volo/d5_512/v5/20220930-080545-volo_d5_512-512/checkpoint-6.pth.tar \
              --output_path ../final_result/volo_d5_512_second_8/ \
              --scoreoutput_path ../final_result/volo_d5_512_second_8/



