# Rewrite ways to test PC effect using DI index

import numpy as np
from scipy import stats
from cnntools import cnntools

from torchvision import models
import torch 
import pandas as pd
from ATT.algorithm import tools
import pickle
from sklearn.decomposition import PCA

def prepare_null_dist(actval, pcamodel, idealmodel, sizerank_pd, dnnrdmdata, n_perm=1000):
    """
    """
    ranklabel = sizerank_pd['real_sizerank'].unique()
    DI_rdm = []
    for i in range(n_perm):
        if i%50 == 0:
            print('Iterate {} times'.format(i))
        dnnrdmdata_flatten = dnnrdmdata.flatten()
        np.random.shuffle(dnnrdmdata_flatten)
        dnnrdmdata = dnnrdmdata_flatten.reshape(*dnnrdmdata.shape)
        pca_rdm = PCA()
        pca_rdm.fit(dnnrdmdata)
        pca_axis_rdm = pca_rdm.components_
        pcacomp_shuffle = np.dot(actval, np.linalg.pinv(pca_axis_rdm))

        avg_ranksize = []
        for lbl in ranklabel:
            avg_ranksize.append(actval[sizerank_pd['real_sizerank']==lbl].mean(axis=0))
        avg_ranksize = np.array(avg_ranksize)
        DI_rdm_eachiter = []
        for pc in range(50):
            actval_delpc = np.dot(np.delete(pcacomp_shuffle, pc, axis=-1), np.delete(pca_axis_rdm, pc, axis=0))
            avg_ranksize_delpc = []
            for lbl in ranklabel:
                avg_ranksize_delpc.append(actval_delpc[sizerank_pd['real_sizerank']==lbl].mean(axis=0))
            avg_ranksize_delpc = np.array(avg_ranksize_delpc)
            DI_rdm_tmp, _, _ = calc_DI(avg_ranksize, avg_ranksize_delpc, idealmodel)
            DI_rdm_eachiter.append(DI_rdm_tmp)
        DI_rdm.append(DI_rdm_eachiter)
    DI_rdm = np.array(DI_rdm)
    return DI_rdm

def calc_DI(actval_rank, actval_rank_delpc, idealmodel):
    """
    """
    r_obj_orig, _ = tools.pearsonr(actval_rank, actval_rank)
    r_obj_delpc, _ = tools.pearsonr(actval_rank_delpc, actval_rank_delpc)
    r_obj_orig_array = r_obj_orig[np.triu_indices(8,1)]
    r_obj_delpc_array = r_obj_delpc[np.triu_indices(8,1)]
    idealmodel_array = idealmodel[np.triu_indices(8,1)]
    r, _ = stats.pearsonr(r_obj_delpc_array, idealmodel_array)
    R, _ = stats.pearsonr(r_obj_orig_array, idealmodel_array)
    DI = 1-(1.0*r/R)
    return DI, r_obj_delpc, r_obj_delpc

if __name__ == '__main__':
    # Load CNN models
    cnn_model = models.alexnet(pretrained=False)
    cnn_model.classifier[-1] = torch.nn.Linear(4096,2)
    cnn_model.load_state_dict(torch.load('/nfs/a1/userhome/huangtaicheng/workingdir/models/DNNmodel_param/alexnetcate2_noaddlayer.pth', map_location='cuda:0'))
    # Load Template
    idealmodel = np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            idealmodel[i,j] = 1-np.abs(i-j)/8
    # Load real-world size file
    sizerank_pd = pd.read_csv('/nfs/a1/userhome/huangtaicheng/workingdir/data/PhysicalSize/Real_SizeRanks8.csv')
    sizerank_pd = sizerank_pd.sort_values('name')
    ranklabel = sizerank_pd['real_sizerank'].unique()
    ranklabel.sort()
    # Load CNN activation
    _, actval = cnntools.extract_activation(cnn_model, '/nfs/a1/userhome/huangtaicheng/workingdir/data/PhysicalSize/ObjectSize/SizeDataset_2021/Object100_origin', layer_loc=('features', '8'), isgpu=True, keeporig=True)
    actval = actval.reshape(*actval.shape[:2], -1).mean(axis=-1)
    actval = actval/np.tile(np.linalg.norm(actval, axis=-1), (actval.shape[-1],1)).T
    # Load original PCA model
    with open('/nfs/a1/userhome/huangtaicheng/workingdir/models/pca_imgnetval_conv4_alexnetcate2.pkl', 'rb') as f:
        pcamodel = pickle.load(f)
    pca_axis = pcamodel.components_
    # Get the averaged size ranks
    avg_ranksize = []
    for lbl in ranklabel:
        avg_ranksize.append(actval[sizerank_pd['real_sizerank']==lbl].mean(axis=0))
    avg_ranksize = np.array(avg_ranksize)
    sizerank_corr, _ = tools.pearsonr(avg_ranksize, avg_ranksize)
    R, _ = stats.pearsonr(sizerank_corr[np.triu_indices(8,1)], idealmodel[np.triu_indices(8,1)])
    # Calculate DI index using normal PCA axis
    pcacomp = np.dot(actval, np.linalg.inv(pca_axis))
    DI = []
    for i in range(50):
        print('PC {}'.format(i+1))
        actval_delpc = np.dot(np.delete(pcacomp, i, axis=-1), np.delete(pca_axis, i, axis=0))
        avg_ranksize_delpc = []
        for lbl in ranklabel:
            avg_ranksize_delpc.append(actval_delpc[sizerank_pd['real_sizerank']==lbl].mean(axis=0))
        avg_ranksize_delpc = np.array(avg_ranksize_delpc)
        DI_tmp, _, _ = calc_DI(avg_ranksize, avg_ranksize_delpc, idealmodel)
        DI.append(DI_tmp)
    DI = np.array(DI)

    # dnn_rdmdata = np.load('/nfs/a1/userhome/huangtaicheng/workingdir/code/PhysicalSize_code/data/inceptionrdm_Mixed6c_branchpool_conv_act_val.npy')
    # dnn_rdmdata = dnn_rdmdata/np.tile(np.linalg.norm(dnn_rdmdata,axis=-1), (dnn_rdmdata.shape[-1],1)).T
    # DI_rdm = prepare_null_dist(actval, pcamodel, idealmodel, sizerank_pd, dnn_rdmdata, n_perm=5000)
    # np.save('/nfs/a1/userhome/huangtaicheng/workingdir/code/PhysicalSize_code/data/DI_nulldist_inception.npy', DI_rdm)



