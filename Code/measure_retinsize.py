# Calculate retinal size of an image

from PIL import Image
from torchvision import transforms
from os.path import join as pjoin
from cnntools import cnntools
import os
import pandas as pd
import numpy as np


class PaddingImage(object):
    """
    """
    def __init__(self, prop):
        self.prop = prop

    def __call__(self,img):
        return cnntools.resize_padding_image(img, prop=self.prop)


parpath = '/nfs/a1/userhome/huangtaicheng/workingdir/data/PhysicalSize/ObjectSize/Object100'
# prop_all = [1.0, 0.8, 0.6, 0.4, 0.2]
prop_all = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
retin_size_all = {}
for prop in prop_all:
    print('Proportion {}'.format(prop))
    img_transform = transforms.Compose([transforms.Resize((224,224)), PaddingImage(prop), transforms.ToTensor()])

    imgcategory = os.listdir(parpath)
    imgcategory.sort()
    retin_size = []

    if 'category_ranksize.csv' in imgcategory:
        imgcategory.pop(imgcategory.index('category_ranksize.csv'))

    for imgs in imgcategory:
        print('Category {}'.format(imgs))
        imgnames = os.listdir(os.path.join(parpath, imgs))
        if '.DS_Store' in imgnames:
            imgnames.pop(imgnames.index('.DS_Store'))
        if 'Thumbs.db' in imgnames:
            imgnames.pop(imgnames.index('Thumbs.db'))

        retin_size_tmp = []
        for imgn in imgnames:
            img = Image.open(pjoin(parpath, imgs, imgn))
            outimg = img_transform(img)
            outimg = outimg.mean(axis=0)
            outimg_obj  = outimg[outimg<0.99]
            retin_size_tmp.append(len(outimg_obj)/(224.0*224.0))
        retin_size.append(np.mean(retin_size_tmp))
    retin_size_all['prop_'+str(prop)] = retin_size
retin_size_all['Name'] = imgcategory
retin_size_all = pd.DataFrame(retin_size_all)
