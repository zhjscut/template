# CUDA_VISIBLE_DEVICES=0,1 python main.py --epochs 50 \
#                 --source_domain infograph --target_domain sketch \
#                 --num_classes 345 --arch resnet50 \
#                 --batch_size 112 --workers 8 --lr 0.01 \
#                 --log log --test_freq 1 --record_freq 1 --print_freq 1 \
#                 --resume log/bs128_lr0.01_ep2500_resnet50_wo_confusion/200_checkpoint.pth.tar  

# CUDA_VISIBLE_DEVICES=2,3 python main.py --epochs 50 \
#                 --source_domain real --target_domain infograph \
#                 --num_classes 345 --arch resnet50 \
#                 --batch_size 112 --workers 8 --lr 0.01 \
#                 --log log --test_freq 1 --record_freq 1 --print_freq 1
               
# CUDA_VISIBLE_DEVICES=12,13 python main.py --epochs 50 \
#                 --source_domain sketch --target_domain quickdraw \
#                 --num_classes 345 --arch resnet50 --pretrained \
#                 --batch_size 112 --workers 8 --lr 0.01 \
#                 --log log --test_freq 1 --record_freq 1 --print_freq 1

CUDA_VISIBLE_DEVICES=14,15 python main.py --epochs 50 \
                --source_domain sketch --target_domain real \
                --num_classes 345 --arch resnet50 --pretrained \
                --batch_size 112 --workers 8 --lr 0.01 \
                --log log --test_freq 1 --record_freq 1 --print_freq 1