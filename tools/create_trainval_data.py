import pickle

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from tools.data_converter import nuscenes_converter as nuscenes_converter


if __name__ == '__main__':
    dataset = 'nuscenes'
    version = 'v1.0'
    train_version = f'{version}-trainval'
    root_path = './data/nuscenes'
    extra_tag = 'bevdetv2-nuscenes'
    nuscenes_version = 'v1.0-trainval'
    saveroot = './'
    nuscenes = NuScenes(nuscenes_version, dataroot)

    train_dataset = pickle.load(
            open('./data/nuscenes/%s_infos_%s.pkl' % (extra_tag, 'train'), 'rb'))
    val_dataset = pickle.load(
            open('./data/nuscenes/%s_infos_%s.pkl' % (extra_tag, 'val'), 'rb'))
    train_dataset['infos'] = train_dataset['infos']+val_dataset['infos']
    with open('./%s_infos_%s.pkl' % (extra_tag, 'trainval'),
                  'wb') as fid:
            pickle.dump(dataset, fid)

    print('add_ann_infos')
    add_ann_adj_info(extra_tag)