export HDF5_USE_FILE_LOCKING='FALSE'
# test multi gpu
python -m torch.distributed.launch --nproc_per_node=4 train_residual.py \
--name residual_scenery_1 \
--train_dataroot /home/v-renhouxing/cn/cc/dataset/scenery/train/ \
--test_dataroot /home/v-renhouxing/cn/cc/dataset/scenery/val/ \
--style_nc 64 --batchSize 18 \
--niter 40 --niter_decay 10 \
--checkpoints_dir ../ckpt/ \
--lr 3e-4 --label_nc 51 \
--no_instance --base_epoch latest \
--init_type "kaiming"  --total_step 10000000 \
--k_mse 0.001 --k_lpips 1 --lmbda 1 \
--dataset_mode ade20k --contain_dontcare_label --with_test \
--non_local \
--binary_quant --qp_step -8 \
--continue_train --which_epoch best

