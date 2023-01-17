**This repo is based on MMDetection v1.0.0rc4**

# environment
## docker
[Dockerfile](docker/Dockerfile)(cuda 11.3,   `docker build -t docker.plusai.co:5050/plusai/mmdetection3d:latest -f ./docker/Dockerfile .`)  


### Pull image
~~~
docker pull docker.plusai.co:5050/plusai/mmdetection3d:latest
~~~

### Run image
~~~
nvidia-docker run  -it  --ipc=host  --mount type=bind,source=/home,target=/home  --mount type=bind,source=/mnt/intel/jupyterhub,target=/mnt/intel/jupyterhub docker.plusai.co:5050/plusai/mmdetection3d /bin/bash 
~~~
## conda
```
conda create --name mm3d-master python=3.8 -y
conda activate mm3d-master

# cuda 10.2
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch -y 
# cuda 11.3
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y 

# following mim installation has nothing to do with cuda version
pip install tensorboardX setuptools==59.5.0 openmim nvitop gitpython

mim install mmcv-full==1.6.1 mmcls==0.23.2 mmdet==2.25.1 mmsegmentation==0.27.0

cd mmdetection3d && pip install -v -e .
```

# train and test work flow
```
git clone git@github-cn.plus.ai:PlusAI/mmdetection3d.git
cd mmdetection3d && pip install -v -e .

# train example
export NCCL_P2P_DISABLE=1 
CUDA_VISIBLE_DEVICES="0,1,2,3" \
PORT=30000 \
bash tools/dist_train.sh \
configs/L3_data_models/pointpillars/pointpillars_L3_vehicle_160e_p6000_pt8_v_025.py \
4 --work-dir work_dirs --extra-tag  pointpillars_L3_exp

```
refer [train.sh](tools/train.sh)     [test.sh](tools/test.sh)

# configs
L3: [configs/L3_data_models](configs/L3_data_models)  
L4: [configs/L4_data_models](configs/L4_data_models)
