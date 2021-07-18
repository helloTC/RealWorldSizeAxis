from cnntools import cnntools
from torchvision import models, transforms
from os.path import join as pjoin
import torch
import numpy as np
import pandas as pd
from scipy import stats, linalg
import os
from dnnbrain.dnn import models as dnn_models
import torch.nn as nn
from PIL import Image
from ATT.iofunc import iofiles
from sklearn.decomposition import PCA
import pickle

def avg_by_imglabel(imgname, actval, label=0):
    """
    """
    lblidx = np.array([imgname[i][1]==label for i in range(len(imgname))])
    return actval[lblidx,:].mean(axis=0)

# Extract PC2
cnn_net = models.alexnet(pretrained=False)
# cnn_net.classifier[-1] = torch.nn.Linear(4096,100)
# cnn_net.classifier = torch.nn.Sequential(*cnn_net.classifier, torch.nn.Linear(1000,2))
# cnn_net.load_state_dict(torch.load('/nfs/a1/userhome/huangtaicheng/workingdir/models/DNNmodel_param/alexnet_twocate.pth'))
# cnn_net.load_state_dict(torch.load('/nfs/a1/userhome/huangtaicheng/workingdir/models/DNNmodel_param/alexnet_object100_singleobj.pth'))
cnn_net.load_state_dict(torch.load('/nfs/a1/userhome/huangtaicheng/workingdir/models/DNNmodel_param/alexnet.pth'))

transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
# transform = transforms.Compose([ShuffleImage(), transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

imgpath_bsobject = '/nfs/a1/userhome/huangtaicheng/workingdir/data/PhysicalSize/ObjectSize/SizeDataset_2021/Object100_origin'
imgname, object_act = cnntools.extract_activation(cnn_net, imgpath_bsobject, layer_loc=('features', '8'), imgtransforms=transform, isgpu=True)

if object_act.ndim == 4:
    object_act = object_act.reshape(*object_act.shape[:2], -1).mean(axis=-1)

object_act_avg = np.zeros((100,object_act.shape[-1]))
for lbl in range(100):
    object_act_avg[lbl,:] = avg_by_imglabel(imgname, object_act, lbl)

object_act_avg = object_act_avg/np.tile(linalg.norm(object_act_avg, axis=1), (object_act_avg.shape[-1],1)).T

iopkl = iofiles.make_ioinstance('/nfs/a1/userhome/huangtaicheng/workingdir/models/pca_imgnetval_conv4_alexnet.pkl')
pca_model = iopkl.load()
# pca_model = PCA()
# pca_model.fit(object_act_avg)

pca_act = np.dot(object_act_avg, np.linalg.pinv(pca_model.components_))
pc2_act = pca_act[:,1]*(-1)

# Load real-world size
# retin_size_pd = pd.read_csv('/nfs/a1/userhome/huangtaicheng/workingdir/data/PhysicalSize/RetinSizes.csv')

rw_size_pd = pd.read_csv('/nfs/a1/userhome/huangtaicheng/workingdir/data/PhysicalSize/Real_SizeRanks8.csv')
rw_size_pd = rw_size_pd.sort_values('name')
with open('/nfs/a1/userhome/huangtaicheng/workingdir/code/PhysicalSize_code/data/xuyitao_1_analysedata.pkl', 'rb') as f:
    rw_size_human = pickle.load(f)

figure_data = {}
figure_data['pc2_act'] = pc2_act
figure_data['sizerank'] = np.array(rw_size_pd['real_sizerank'])
figure_data['rw_size_human'] = rw_size_human['subjsize_rank']
figure_data = pd.DataFrame(figure_data)

import matplotlib.cm as cm
import matplotlib.pyplot as plt
cmap = plt.get_cmap('rainbow')
for rank in np.arange(1,9,1):
    rank_data = figure_data[figure_data['sizerank']==rank]
    plt.scatter(rank_data['pc2_act'], rank_data['rw_size_human'], color=cmap(rank/9))

# plt.legend(['SizeRank '+str(rank) for rank in np.arange(1,9,1)])
# plt.show()
    






