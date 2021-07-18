# Examine effect of each PC after remove one but kept the others

import numpy as np
from scipy import stats
from ATT.iofunc import iofiles
from cnntools import cnntools

from torchvision import models
from torch import nn
import torch
import pandas as pd
from ATT.algorithm import tools

from sklearn.decomposition import PCA

cnn_model = models.alexnet(pretrained=False)
cnn_model.features = torch.nn.DataParallel(cnn_model.features)
cnn_model.cuda()
checkpoint = torch.load('/nfs/a1/userhome/huangtaicheng/workingdir/models/DNNmodel_param/alexnet_shapebiased.pth.tar')
cnn_model.load_state_dict(checkpoint["state_dict"])

sizerank_pd = pd.read_csv('/nfs/a1/userhome/huangtaicheng/workingdir/data/PhysicalSize/Real_SizeRanks8.csv')
sizerank_pd = sizerank_pd.sort_values('name')
ranklabel = sizerank_pd['real_sizerank'].unique()
ranklabel.sort()

imgnames, actval = cnntools.extract_activation(cnn_model, '/nfs/a1/userhome/huangtaicheng/workingdir/data/PhysicalSize/ObjectSize/SizeDataset_2021/Object100_origin', layer_loc=('features', 'module', '6'), isgpu=True)
actval = actval.reshape(*actval.shape[:2], -1).mean(axis=-1)
actval = actval/np.tile(np.linalg.norm(actval,axis=-1), (actval.shape[-1],1)).T

iopkl = iofiles.make_ioinstance('/nfs/a1/userhome/huangtaicheng/workingdir/models/pca_imgnetval_conv3_alexnetshape.pkl')
pcamodel = iopkl.load()

pcacomp = np.dot(actval, np.linalg.inv(pcamodel.components_))

# Template
real_temp = np.zeros((8,8))
for i in range(8):
    for j in range(8):
        real_temp[i,j] = 1-np.abs(i-j)/8

avg_ranksize = []
for lbl in ranklabel:
    avg_ranksize.append(actval[sizerank_pd['real_sizerank']==lbl].mean(axis=0))
avg_ranksize = np.array(avg_ranksize)
r_obj, _ = tools.pearsonr(avg_ranksize, avg_ranksize)
r_realsize_baseline, _ = stats.pearsonr(r_obj[np.triu_indices(8,1)], real_temp[np.triu_indices(8,1)])

r_realsize_delpc = []
r_obj_delpc_all = []
for i in range(50):
    actval_delpc = np.dot(np.delete(pcacomp, i, axis=-1), np.delete(pcamodel.components_, i, axis=0))
    avg_ranksize_delpc = []
    for lbl in ranklabel:
        avg_ranksize_delpc.append(actval_delpc[sizerank_pd['real_sizerank']==lbl].mean(axis=0))
    avg_ranksize = np.array(avg_ranksize)
    r_obj_delpc, _ = tools.pearsonr(avg_ranksize_delpc, avg_ranksize_delpc)
    r_obj_delpc_all.append(r_obj_delpc)
    r_realsize_delpc.append(stats.pearsonr(r_obj_delpc[np.triu_indices(8,1)], real_temp[np.triu_indices(8,1)])[0])
r_realsize_delpc = np.array(r_realsize_delpc)
r_obj_delpc_all = np.array(r_obj_delpc_all)
DI = 1-(r_realsize_delpc/r_realsize_baseline)




