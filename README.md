## Set the env
```
conda create -n vsr_llm python=3.10.12 -y
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
conda install ffmpeg==7.1.0
```

## [preparation](https://github.com/mpc001/auto_avsr/tree/main/preparation)
process the dataset from auto_avsr