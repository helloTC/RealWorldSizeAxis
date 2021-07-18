# build alexnet and extract activation of it.
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
import numpy as np
import pickle
from sklearn.decomposition import PCA

def rm_pc(actval, pcamodel, pc=1):
    """
    """
    actval = actval.detach().numpy()
    actval_flat = actval.reshape(*actval.shape[:2], actval.shape[2]*actval.shape[3])
    actval_avg = actval_flat.mean(axis=-1)
    norm_actval = np.linalg.norm(actval_avg, axis=-1)
    norm_actval = np.tile(norm_actval[:,None,None], (1,actval_flat.shape[1],actval_flat.shape[2]))
    actval_flat_norm = actval_flat/norm_actval
    # pca axis
    pca_axis = pcamodel.components_
    pca_axis_inv = np.linalg.inv(pca_axis)
    actval_flat_delpc = np.zeros_like(actval_flat)
    for i in range(actval_flat.shape[-1]):
        pcacomp_tmp = np.dot(actval_flat_norm[...,i], pca_axis_inv)
        # Remove PC
        actval_flat_delpc_tmp = np.dot(np.delete(pcacomp_tmp, pc, axis=-1), np.delete(pca_axis, pc, axis=0))
        # Test: Remove first n PCs
        # actval_flat_delpc_tmp = np.dot(np.delete(pcacomp_tmp, np.arange(pc+1), axis=-1), np.delete(pca_axis, np.arange(pc+1), axis=0))
        actval_flat_delpc[...,i] = actval_flat_delpc_tmp
    actval_flat_delpc = actval_flat_delpc*norm_actval
    actval_delpc = actval_flat_delpc.reshape(*actval.shape)
    actval_delpc = torch.Tensor(actval_delpc)
    return actval_delpc

def prepare_null_distribution_acc(alexnet, data, target, corr_num, pc=1, n_iter=1000):
    """
    """
    # Load random data for the generation of PCA model
    rdmdata = np.load('/nfs/a1/userhome/huangtaicheng/workingdir/code/PhysicalSize_code/data/alexnetrdm_conv4_act_val.npy')
    # Prepare alexnet
    alexnet_seq = nn.Sequential(*list(alexnet.children())).eval()
    alexnet_conv4 = alexnet_seq[0][:9]
    alexnet_conv7 = alexnet_seq[0][9:]
    alexnet_avgpool = alexnet_seq[1].eval()
    alexnet_linear = alexnet_seq[2].eval()
    # Load data
    outdata_conv4 = alexnet_conv4(data)
    for n in range(n_iter):
        print(' Iteration {}'.format(n+1))
        # Shuffle rdm data to increase uncertainty of the data
        rdmdata_flatten = rdmdata.flatten()
        np.random.shuffle(rdmdata_flatten)
        rdmdata = rdmdata_flatten.reshape(*rdmdata.shape)
        # prepare pcamodel in random situation.
        pcamodel_rdm = PCA()
        pcamodel_rdm.fit(rdmdata)
        outdata_conv4_tmp = rm_pc(outdata_conv4, pcamodel_rdm, pc=pc)
        # Transfer outdata into further layers
        outdata_tmp = alexnet_conv7(outdata_conv4_tmp)
        outdata_tmp = alexnet_avgpool(outdata_tmp)
        outdata_tmp = torch.flatten(outdata_tmp, 1)
        outdata_tmp = alexnet_linear(outdata_tmp)
        outdata_tmp_np = outdata_tmp.detach().numpy()
        corr_num[n] += sum(outdata_tmp_np.argmax(axis=-1)==np.array(target))
    return corr_num

def reorganize_alexnet(alexnet, data, pcamodel=None, pc=1):
    """
    """
    alexnet_seq = nn.Sequential(*list(alexnet.children())).eval()
    alexnet_conv4 = alexnet_seq[0][:9]
    alexnet_conv7 = alexnet_seq[0][9:]
    alexnet_avgpool = alexnet_seq[1].eval()
    alexnet_linear = alexnet_seq[2].eval()
    # Load data
    outdata_conv4 = alexnet_conv4(data)
    if pcamodel is not None:
        outdata_conv4 = rm_pc(outdata_conv4, pcamodel, pc=pc)
    outdata_conv4_input = 1.0*outdata_conv4
    outdata = alexnet_conv7(outdata_conv4_input)
    outdata = alexnet_avgpool(outdata)
    outdata = torch.flatten(outdata, 1)
    outdata = alexnet_linear(outdata)
    return outdata_conv4, outdata

def calc_topk(actval, target, k=1):
    """
    """
    isacc = []
    for i in range(actval.shape[0]):
        act = actval[i,:]
        idx = np.argsort(act)[-k:]
        if target[i] in idx:
            isacc.append(True)
        else:
            isacc.append(False)
    return sum(isacc)/len(isacc)

if __name__ == '__main__':
    alexnet = models.alexnet(pretrained=False)
    # alexnet.classifier[-1] = torch.nn.Linear(4096,100)
    alexnet.load_state_dict(torch.load('/nfs/a1/userhome/huangtaicheng/workingdir/models/DNNmodel_param/alexnet.pth', map_location='cpu'))
    alexnet = alexnet.eval()

    # Load PCA model
    with open('/nfs/a1/userhome/huangtaicheng/workingdir/models/pca_imgnetval_conv4_alexnet.pkl', 'rb') as f:
        pcamodel = pickle.load(f)

    img_transform = transforms.Compose([transforms.Resize((224,224)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    imgpath = '/nfs/a1/ImgDatabase/ImageNet_2012/ILSVRC2012_img_val'
    # imgpath = '/nfs/a1/userhome/huangtaicheng/workingdir/data/PhysicalSize/ObjectSize/Object100'
    imagefolder = datasets.ImageFolder(imgpath, img_transform)
    dataloader = torch.utils.data.DataLoader(imagefolder, batch_size=50, num_workers=10, shuffle=False)
   
    actval_all_rmpc = np.zeros((50000,1000))
    actval_all = np.zeros((50000, 1000))
    target = []
    for i in range(1000):
        target.extend([i]*50)
    target = np.array(target)

    # n_iter = 2
    # corr_num = np.zeros((n_iter, 1))
    for i, (img, _) in enumerate(dataloader):
        print('Image {}:{}'.format(50*i,50*(i+1)))
        actval_conv4_rmpc, actval_rmpc = reorganize_alexnet(alexnet, img, pcamodel=pcamodel, pc=1)
        actval_conv4, actval = reorganize_alexnet(alexnet, img, pcamodel=None)
        actval_all_rmpc[50*i:50*(i+1), :] = actval_rmpc.detach().numpy()
        actval_all[50*i:50*(i+1), :] = actval.detach().numpy()
        # break
        # corr_num = prepare_null_distribution_acc(alexnet, img, target, corr_num, pc=1, n_iter=n_iter)
    # acc_rdm = corr_num/50000.0
    # np.save('alexnet_rmpc_nulldist_acc.npy', acc_rdm)
    # acc = calc_topk(actval_all, target, k=1)
    acc = []
    acc_rmpc = []
    for i in range(1000):
        acc.append(calc_topk(actval_all[i*50:(i+1)*50,:], [i]*50, k=1))
        acc_rmpc.append(calc_topk(actval_all_rmpc[i*50:(i+1)*50,:], [i]*50, k=1))
    acc = np.array(acc)
    acc_rmpc = np.array(acc_rmpc)



