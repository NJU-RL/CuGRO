#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8  main-gene.py --env "cheetah_vel" --data_mode "gene" --actor_type "large" --diffusion_steps 100
python critic.py --env "cheetah_vel" --data_mode "gene" --actor_type "large" --diffusion_steps 100 --gpu 0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8   main-gene.py --env "swimmer_dir" --data_mode "gene" --actor_type "large" --diffusion_steps 100
python critic.py --env "swimmer_dir" --data_mode "gene" --actor_type "large" --diffusion_steps 100 --gpu 0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8  main-gene.py --env "walker_params" --data_mode "gene" --actor_type "large" --diffusion_steps 100
python critic.py --env "walker_params" --data_mode "gene" --actor_type "large" --diffusion_steps 100  --gpu 0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 main-gene.py --env "meta_world" --data_mode "gene" --actor_type "large" --diffusion_steps 100
python critic.py --env "meta_world" --data_mode "gene" --actor_type "large" --diffusion_steps 100 --gpu 0

