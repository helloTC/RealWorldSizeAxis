# Calculate Big Small object/animate thing difference

from cnntools import cnntools
from ATT.algorithm import tools
from scipy import stats, linalg
import numpy as np
from torchvision import models, transforms
import torch
from scipy import stats
from ATT.iofunc import iofiles
import pandas as pd
import os


def avg_by_imglabel(imgname, actval, label=0):
    """
    """
    lblidx = np.array([imgname[i][1]==label for i in range(len(imgname))])
    return actval[lblidx,:].mean(axis=0)



cnn_net = models.resnet34(pretrained=False)
# cnn_net.classifier[-1] = torch.nn.Linear(4096,100)
# cnn_net.classifier = torch.nn.Sequential(*cnn_net.classifier, torch.nn.Linear(1000,2))
# cnn_net.load_state_dict(torch.load('/nfs/a1/userhome/huangtaicheng/workingdir/models/DNNmodel_param/alexnet_object100_singleobj.pth'))
cnn_net.load_state_dict(torch.load('/nfs/a1/userhome/huangtaicheng/workingdir/models/DNNmodel_param/resnet34.pth'))
# cnn_net.load_state_dict(torch.load('/nfs/a1/userhome/huangtaicheng/workingdir/models/DNNmodel_param/alexnet_twocate.pth'))

transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

sizerank_pd = pd.read_csv('/nfs/a1/userhome/huangtaicheng/workingdir/data/PhysicalSize/Real_SizeRanks8.csv')
sizerank_pd = sizerank_pd.sort_values('name')
ranklabel = sizerank_pd['real_sizerank'].unique()
ranklabel.sort()

real_temp = np.zeros((len(ranklabel), len(ranklabel)))
for i in range(len(ranklabel)):
    for j in range(len(ranklabel)):
        real_temp[i,j] = 1-np.abs(i-j)/len(ranklabel)

# real_temp = np.load('../../data/human_labelled_sizerank.npy')

imgpath_bsobject = '/nfs/a1/userhome/huangtaicheng/workingdir/data/PhysicalSize/ObjectSize/SizeDataset_2021/Object100_origin'

layer_loc_all = [('conv1'), ('layer1', '0', 'conv1'),
                 ('layer1', '0', 'conv2'), ('layer1', '1', 'conv1'),
                 ('layer1', '1', 'conv2'), 
                 ('layer1', '2', 'conv1'), ('layer1', '2', 'conv2'),
                 ('layer2', '0', 'conv1'), ('layer2', '0', 'conv2'),
                 ('layer2', '0', 'downsample', '0'), ('layer2', '1', 'conv1'),
                 ('layer2', '1', 'conv2'), 
                 ('layer2', '2', 'conv1'), ('layer2', '2', 'conv2'),
                 ('layer2', '3', 'conv1'), ('layer2', '3', 'conv2'),
                 ('layer3', '0', 'conv1'), ('layer3', '0', 'conv2'), 
                 ('layer3', '0', 'downsample', '0'),
                 ('layer3', '1', 'conv1'), ('layer3', '1', 'conv2'),
                 ('layer3', '2', 'conv1'), ('layer3', '2', 'conv2'),
                 ('layer3', '3', 'conv1'), ('layer3', '3', 'conv2'),
                 ('layer3', '4', 'conv1'), ('layer3', '4', 'conv2'),
                 ('layer3', '5', 'conv1'), ('layer3', '5', 'conv2'),
                 ('layer4', '0', 'conv1'),
                 ('layer4', '0', 'conv2'),
                 ('layer4', '0', 'downsample', '0'), ('layer4', '1', 'conv1'),
                 ('layer4', '1', 'conv2'), 
                 ('layer4', '2', 'conv1'), ('layer4', '2', 'conv2'), 
                 ('fc')]

r_obj_all = []
sizepref = []
for layer_loc in layer_loc_all:
    print('layer loc: {}'.format(layer_loc))
    imgname, object_act = cnntools.extract_activation(cnn_net, imgpath_bsobject, layer_loc=layer_loc, isgpu=True, imgtransforms=transform)
    if object_act.ndim > 2:
        object_act = object_act.reshape(*object_act.shape[:2],-1).mean(axis=-1)
    object_act = object_act/np.tile(np.linalg.norm(object_act,axis=-1), (object_act.shape[-1],1)).T
    avg_ranksize = []
    for lbl in ranklabel:
        avg_ranksize.append(object_act[sizerank_pd['real_sizerank']==lbl].mean(axis=0))
    avg_ranksize = np.array(avg_ranksize)
    # avg_ranksize = avg_ranksize/np.tile(np.linalg.norm(avg_ranksize,axis=1), (avg_ranksize.shape[-1],1)).T

    r_obj, _ = tools.pearsonr(avg_ranksize, avg_ranksize)
    r_obj_all.append(r_obj)

    sizepref_tmp, _ = stats.pearsonr(r_obj[np.triu_indices(8,1)], real_temp[np.triu_indices(8,1)])
    sizepref.append(sizepref_tmp)
r_obj_all = np.array(r_obj_all)
sizepref = np.array(sizepref)


# iopkl = iofiles.make_ioinstance('../../models/pca_imgnetval_conv4_alexnet.pkl')
# iopkl = iofiles.make_ioinstance('../../models/pca_liuzmvid_train.pkl')
# pca_model = iopkl.load()
# ranksize_comp = pca_model.transform(avg_ranksize)



