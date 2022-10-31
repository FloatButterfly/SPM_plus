export HDF5_USE_FILE_LOCKING='FALSE'
# test multi gpu
CUDA_VISIBLE_DEVICES=0,1,2,3, \
python -m torch.distributed.launch --nproc_per_node=4 train_sean_vid.py \
--name ravdess_recon \
--train_img_csv /home/v-renhouxing/cn/cc/dataset/ravdess/256/train_pairs_imgs.csv \
--train_lab_csv /home/v-renhouxing/cn/cc/dataset/ravdess/256/train_pairs_labs.csv \
--test_img_csv /home/v-renhouxing/cn/cc/dataset/ravdess/256/val_pairs_imgs.csv \
--test_lab_csv /home/v-renhouxing/cn/cc/dataset/ravdess/256/val_pairs_labs.csv \
--style_nc 64 --batchSize 16 \
--niter 40 --niter_decay 20 \
--checkpoints_dir ../ckpt/vid_ravdess/ \
--lr 3e-4 --no_TTUR --label_nc 18 \
--no_instance --continue_train --base_epoch 6 \
--init_type "kaiming"  --total_step 10000000 \
--k_latent 0 --k_latent2 0 --k_cd 0 --k_L1 0 --k_latent_reg 0 --k_recon_reg 1 \
--k_feat 10 --k_gan 1 --k_vgg 10 --k_latent_reg2 1 --lmbda 0 \
--dataset_mode celeba --contain_dontcare_label --with_test \
--ds_label \
--qp_step -8 

# --labels_x8
# --model sean \