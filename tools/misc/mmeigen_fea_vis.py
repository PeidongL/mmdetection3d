import numpy as np
import os
import torch
import mmcv
from mmengine.visualization import Visualizer
from torchvision.models import resnet50
from torchvision.transforms import Compose, Normalize, ToTensor
import torch.nn.functional as F

def preprocess_image(img, mean, std):
    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)

model = resnet50(pretrained=True)

def _forward(x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x1 = model.layer1(x)
    x2 = model.layer2(x1)
    x3 = model.layer3(x2)
    x4 = model.layer4(x3)
    return x4

def resize_feature(out_h, out_w, in_feat):
    new_h = torch.linspace(-1, 1, out_h).view(-1, 1).repeat(1, out_w)
    new_w = torch.linspace(-1, 1, out_w).repeat(out_h, 1)
    grid = torch.cat((new_w.unsqueeze(2), new_h.unsqueeze(2)), dim=2)
    grid = grid.unsqueeze(0)
    grid = grid.expand(in_feat.shape[0], *grid.shape[1:]).to(in_feat)
    
    out_feat = F.grid_sample(in_feat, grid=grid, mode='bilinear', align_corners=False)
    
    return out_feat

model.forward = _forward

root_path='/mnt/intel/jupyterhub/swc/datasets/L4E_extracted_data_1227/L4E_origin_data/training'
idx='000000.1612512282.899542.20210205T141034_j7-feidian_25_132to152'

cam_name='side_left_camera'
cam_name='front_left_camera'
# cam_name='side_right_camera'

img_name= os.path.join(root_path, cam_name+'/'+idx+'.jpg')
feat_name=os.path.join(root_path, cam_name+'_py_feature'+'/'+idx+'_0.npy')
image = mmcv.imread(img_name, channel_order='rgb')
feat=np.load(feat_name)

visualizer = Visualizer(image=image)
# image_norm = np.float32(image) / 255
# input_tensor = preprocess_image(image_norm,
#                                 mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])
# feat = model(input_tensor)[0]

feat=torch.Tensor(feat)
feat=feat.squeeze(0)
visualizer = Visualizer()


# sum_channel_featmap = torch.sum(feat, dim=(1, 2))
# print('aaaaaaaa', sum_channel_featmap.shape)

# _, indices = torch.topk(sum_channel_featmap, 1)
# feat_map = feat[indices]

feat = F.interpolate(
    feat[None],
    (45, 80),
    mode='bilinear',
    align_corners=False)[0]

# feat = resize_feature(45,80,feat[None])[0]

# drawn_img = visualizer.draw_featmap(feat, image, channel_reduction='select_max')
idx=9
# drawn_img = visualizer.draw_featmap(feat, channel_reduction='select_max')
# drawn_img = visualizer.draw_featmap(feat[idx:idx+1], channel_reduction='select_max')
# drawn_img = visualizer.draw_featmap(torch.Tensor(image.transpose(2,0,1)), image, topk=-1, channel_reduction=None)
drawn_img = visualizer.draw_featmap(feat, image, channel_reduction='select_max')

# drawn_img = visualizer.draw_featmap(feat, image, channel_reduction=None, topk=64,arrangement=(8, 8))
visualizer.show(drawn_img)

# 参考https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/visualization.html