#!/bin/bash
iters="340000_head.tar"
names="cnn"
datasets="cnn"
near=0.5858861088752747
far=1.1858861088752746
path="dataset/finetune_models/${names}/${iters}"
datapath="dataset/${datasets}/0"
bc_type="torso_imgs"
suffix="val"
python NeRFs/run_nerf_deform.py --need_torso True --config dataset/test_config.txt --expname ${names}_${suffix} --expname_finetune ${names}_${suffix} --render_only --ft_path ${path} --datadir ${datapath} --bc_type ${bc_type} --near ${near} --far ${far}

