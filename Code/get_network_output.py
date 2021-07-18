import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models as tv_models
from torchvision import transforms
from cnntools import cnntools
from cnntools import transforms as T
from ATT.algorithm import tools
from os.path import join as pjoin

# Load model
cnn_net = tv_models.alexnet(pretrained=False)
cnn_net.classifier[-1] = torch.nn.Linear(4096,2)
# cnn_net.classifier = torch.nn.Sequential(*cnn_net.classifier, torch.nn.Linear(1000,2))
# cnn_net.load_state_dict(torch.load('/nfs/a1/userhome/huangtaicheng/workingdir/models/DNNmodel_param/alexnet_twocate.pth'))
cnn_net.load_state_dict(torch.load('/nfs/a1/userhome/huangtaicheng/workingdir/models/DNNmodel_param/alexnetcate2_noaddlayer.pth', map_location='cuda:0'))
# Transforms
img_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

imgpath = '/nfs/a1/ImgDatabase/ImageNet_2012/ILSVRC2012_img_val'
# imgpath = '/nfs/a1/userhome/huangtaicheng/workingdir/data/ImageNet_boundingbox/ILSVRC1000_val_thrimg'
_, actval = cnntools.extract_activation(cnn_net, imgpath, layer_loc=('features', '8'), imgtransforms=img_transform, isgpu=True, batch_size=50)

# if actval.ndim > 2:
#     actval = actval.reshape(*actval.shape[:2],-1).mean(axis=-1)

# actval_avg = np.zeros((1000,actval.shape[-1]))
# for i in range(1000):
#     actval_avg[i,...] = actval[50*i:50*(i+1), :].mean(axis=0)

np.save('/nfs/a1/userhome/huangtaicheng/workingdir/code/PhysicalSize_code/data/alexnetcate2_conv4_act_val.npy', actval)


