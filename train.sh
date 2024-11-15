export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,3
# IF the GPUS are 4000 series, use the following command
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=1
# run torchrun for DDP
# torchrun --nproc_per_node 2 src/train.py
# run accelerate launch for DDP
accelerate launch src/train.py