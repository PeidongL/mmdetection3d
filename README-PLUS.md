# This repo is based on MMDetection v1.0.0rc4

conda create --name mm3d-master python=3.8 -y
conda activate mm3d-master

# cuda 10.2
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
# cuda 11.3
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge


pip install tensorboardX 
pip install setuptools==59.5.0

# following mim installation has nothing to do with cuda version
pip install openmim
mim install mmcv-full==1.6.1
mim install mmcls==0.23.2
mim install mmdet==2.25.1
mim install mmsegmentation==0.27.0


cd mmdetection3d && pip install -v -e .
