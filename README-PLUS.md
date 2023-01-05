**This repo is based on MMDetection v1.0.0rc4**

# environment
## docker
[Dockerfile](docker/Dockerfile) (cuda 11.3)

1. build docker: `docker build -t mm3d:develop -f ./docker/Dockerfile .`

2. run docker: 
` 
nvidia-docker run  -it --name mm3d_develop --ipc=host  --mount type=bind,source=/home,target=/home  --mount type=bind,source=/mnt/intel/jupyterhub,target=/mnt/intel/jupyterhub mm3d:develop /bin/bash
`
3. `cd mmdetection3d && pip install -v -e .`

## conda
```
conda create --name mm3d-master python=3.8 -y
conda activate mm3d-master

# cuda 10.2
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
# cuda 11.3
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

# following mim installation has nothing to do with cuda version
pip install tensorboardX setuptools==59.5.0 openmim nvitop

mim install mmcv-full==1.6.1 mmcls==0.23.2 mmdet==2.25.1 mmsegmentation==0.27.0

cd mmdetection3d && pip install -v -e .
```

# train and test
refer [train.sh](tools/train.sh)     [test.sh](tools/test.sh)

# configs
L3: [configs/L3_data_models](configs/L3_data_models)  
L4: [configs/L4_data_models](configs/L4_data_models)
