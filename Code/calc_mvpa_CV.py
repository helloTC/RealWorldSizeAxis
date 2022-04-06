# Extract activation values
from os.path import join as pjoin
import nibabel as nib
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA, FastICA
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.svm import SVC

def _get_values(subjid, taskcond, cope, mask, parpath=None):
    """
    """
    cope_path = pjoin(parpath, subjid, 'MNINonLinear', 'Results', taskcond, taskcond+'_hp200_s4_level2.feat/GrayordinatesStats', cope+'.feat', 'tstat1.dtseries.nii')
    cope_act = nib.load(cope_path).get_data()[0,:59412]
    cope_act = cope_act[mask]
    return cope_act

def pattern_corr(subjid_all, taskcond1, taskcond2, cope1, cope2, mask, parpath=None):
    """
    """
    r_all = []
    for subjid in subjid_all:
        cope1_act = _get_values(subjid, taskcond1, cope1, mask, parpath=None)
        cope2_act = _get_values(subjid, taskcond2, cope2, mask, parpath=None)
        r, _ = stats.pearsonr(cope1_act, cope2_act)
        r_all.append(r)
    r_all = np.array(r_all)
    return r_all

parpath = '/home/hellotc/workdir/fMRI/SizePreference/pretest/nifti/derivatives/ciftify/'

mask = nib.load('/home/hellotc/workdir/test/palm_test/origsize_mask_tval.dscalar.nii').get_data()
mask = mask[0,:59412]
mask_bin = (mask!=0)

subjid_all = ['sub-S'+str(subjid).zfill(2) for subjid in np.arange(1,11,1)]

bigorig_values = []
smallorig_values = []

bigsketch_values = []
smallsketch_values = []

bigtexture_values = []
smalltexture_values = []

for subjid in subjid_all:
    bigorig_values.append(_get_values(subjid, 'task-size-odds', 'cope1', mask_bin, parpath))
    smallorig_values.append(_get_values(subjid, 'task-size-odds', 'cope2', mask_bin, parpath))
    bigsketch_values.append(_get_values(subjid, 'task-size-odds', 'cope4', mask_bin, parpath))
    smallsketch_values.append(_get_values(subjid, 'task-size-odds', 'cope5', mask_bin, parpath))
    bigtexture_values.append(_get_values(subjid, 'task-size-odds', 'cope7', mask_bin, parpath))
    smalltexture_values.append(_get_values(subjid, 'task-size-odds', 'cope8', mask_bin, parpath))

bigorig_values = np.array(bigorig_values).T
smallorig_values = np.array(smallorig_values).T
bigsketch_values = np.array(bigsketch_values).T
smallsketch_values = np.array(smallsketch_values).T
bigtexture_values = np.array(bigtexture_values).T
smalltexture_values = np.array(smalltexture_values).T

labels = np.array([1]*10 + [2]*10)
orig = np.concatenate((bigorig_values, smallorig_values), axis=-1)
orig = stats.zscore(orig, axis=-1)
sketch = np.concatenate((bigsketch_values, smallsketch_values), axis=-1)
sketch = stats.zscore(sketch, axis=-1)
texture = np.concatenate((bigtexture_values, smalltexture_values), axis=-1)
texture = stats.zscore(texture, axis=-1)

# pca = PCA()
# orig_pc = pca.fit_transform(orig.T)
# sketch_pc = pca.fit_transform(sketch.T)
# texture_pc = pca.fit_transform(texture.T)

clf = SVC(kernel='linear')
cv = model_selection.LeaveOneOut()
# cv = model_selection.StratifiedKFold(2, shuffle=True)
score_orig, perm_scores, pvalue = model_selection.permutation_test_score(clf, orig.T, labels, scoring='accuracy', cv=cv, n_permutations=10000)
score_sketch, _, _ = model_selection.permutation_test_score(clf, sketch.T, labels, scoring='accuracy', cv=cv, n_permutations=1)
score_texture, _, _ = model_selection.permutation_test_score(clf, texture.T, labels, scoring='accuracy', cv=cv, n_permutations=1)

# import matplotlib.pyplot as plt

# plt.hist(perm_scores, 50, color='gray')
# plt.savefig('mvpa_hist.tif', dpi=300)


