## Set the env
```
conda create -n vsr_llm python=3.10.12 -y
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
conda install ffmpeg==7.1.0
python -m pip install --upgrade pip==23.2 (from 24.3.1)
pip install hydra-core --upgrade (it will occur error but ignore)
```

## [preparation](https://github.com/mpc001/auto_avsr/tree/main/preparation)
process the dataset from auto_avsr

## Note 
1. The cfg.dataset.max_frams is for total_gpus not only for one per 4090 I suggest the value is 600~800