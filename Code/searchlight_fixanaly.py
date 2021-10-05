# A streamline do searchlight analysis across participants
# Streamline included several steps below.
# 1. Seperate cifti t map from each subject into metrics in left/right hemispheres
# 2. Do searchlight analysis, and save it.
# 3. Merge correlations in left/right hemispheres together.
# 4. Extract the averaged correlation by using the mask. 

import numpy as np
from os.path import join as pjoin
from ATT.algorithm import surf_tools, tools
import subprocess
import nibabel as nib
from scipy import stats
from ATT.iofunc import iofiles
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import model_selection

data_parpath = '/home/hellotc/workdir/fMRI/SizePreference/pretest/nifti/derivatives/ciftify/'

outdata_parpath = '/home/hellotc/workdir/test/tmp'

def _cifti_separate(subject, copes):
    """
    """
    cifti_in_path = pjoin(data_parpath, subject, 'MNINonLinear', 'Results', 'task-size-all', 'task-size-all_hp200_s4_level2.feat', 'GrayordinatesStats', copes+'.feat', 'tstat1.dtseries.nii')
    separate_command_list = ['wb_command', '-cifti-separate', cifti_in_path, 'COLUMN', '-metric', 'CORTEX_LEFT', pjoin(outdata_parpath, subject+'_'+copes+'_left.func.gii'), '-metric', 'CORTEX_RIGHT', pjoin(outdata_parpath, subject+'_'+copes+'_right.func.gii')]
    subprocess.call(' '.join(separate_command_list), shell=True)

def cifti_separate(subject):
    """
    """
    print('Now separate subject {}'.format(subject))
    _cifti_separate(subject, 'cope7')
    _cifti_separate(subject, 'cope8')


def searchlight(subjid_all, n=5):
    """
    """
    # Prepare data
    geometry_lh = nib.load(pjoin(outdata_parpath, 'L_midthickness.surf.gii')).darrays[1].data
    geometry_rh = nib.load(pjoin(outdata_parpath, 'R_midthickness.surf.gii')).darrays[1].data

    bigvalue_left = np.zeros((10,32492))
    bigvalue_right = np.zeros((10,32492))
    smallvalue_left = np.zeros((10,32492))
    smallvalue_right = np.zeros((10,32492)) 

    for i, subjid in enumerate(subjid_all):
        bigvalue_left[i,:] = nib.load(pjoin('tmp', subjid+'_cope7_left.func.gii')).darrays[0].data
        smallvalue_left[i,:] = nib.load(pjoin('tmp', subjid+'_cope8_left.func.gii')).darrays[0].data
        bigvalue_right[i,:] = nib.load(pjoin('tmp', subjid+'_cope7_right.func.gii')).darrays[0].data
        smallvalue_right[i,:] = nib.load(pjoin('tmp', subjid+'_cope8_right.func.gii')).darrays[0].data

    labels = np.array([1]*10 + [2]*10)
    value_left = np.concatenate((bigvalue_left, smallvalue_left),axis=0)
    value_right = np.concatenate((bigvalue_right, smallvalue_right),axis=0)

    clf = SVC(kernel='linear')
    cv = model_selection.StratifiedKFold(2, shuffle=True)
    
    pca = PCA()

    score_left = np.zeros((32492))
    score_right = np.zeros((32492))
    pvalue_left = np.zeros((32492))
    pvalue_right = np.zeros((32492))

    for i in range(32492):
        if i%1000 == 0:
            print('Searchlight for vx {}'.format(i+1))
        vxlist_left = surf_tools.get_n_ring_neighbor(i, geometry_lh, n)[0]
        vxlist_left = np.array(list(vxlist_left))
        vxlist_right = surf_tools.get_n_ring_neighbor(i, geometry_lh, n)[0]
        vxlist_right = np.array(list(vxlist_right))
        
        value_left_tmp = value_left[:,vxlist_left]
        value_left_tmp = stats.zscore(value_left_tmp, axis=0)
        value_left_tmp[np.isnan(value_left_tmp)] = 0
        value_right_tmp = value_right[:,vxlist_right]
        value_right_tmp = stats.zscore(value_right_tmp, axis=0)
        value_right_tmp[np.isnan(value_right_tmp)] = 0
        if (0 in value_left_tmp) or (0 in value_right_tmp):
            continue

        # value_left_pca = pca.fit_transform(value_left_tmp)
        # value_right_pca = pca.fit_transform(value_right_tmp)

        # score_left_tmp, _, pvalue_left_tmp = model_selection.permutation_test_score(clf, value_left_pca, labels, scoring='accuracy', cv=cv, n_permutations=1)
        score_left_tmp = model_selection.cross_val_score(clf, value_left_tmp, labels, scoring='accuracy', cv=cv)
        if score_left_tmp.mean()>0.75:
            print('Score left hemi {}'.format(score_left_tmp.mean()))
        # score_right_tmp, _, pvalue_right_tmp = model_selection.permutation_test_score(clf, value_right_pca, labels, scoring='accuracy', cv=cv, n_permutations=1)
        score_right_tmp = model_selection.cross_val_score(clf, value_right_tmp, labels, scoring='accuracy', cv=cv)
        if score_right_tmp.mean()>0.75:
            print('Score right hemi {}'.format(score_right_tmp.mean()))
        score_left[i] = score_left_tmp.mean()
        score_right[i] = score_right_tmp.mean()
    iogii_score_left = iofiles.make_ioinstance(pjoin('tmp', 'texture_score_left_nopc_n5.func.gii'))
    iogii_score_right = iofiles.make_ioinstance(pjoin('tmp', 'texture_score_right_nopc_n5.func.gii'))
    iogii_score_left.save(score_left, 'CortexLeft')
    iogii_score_right.save(score_right, 'CortexRight')
 
def cifti_merge():
    """
    """
    print('Merge into CIFTI')
    merge_command_list_orig = ['wb_command', '-cifti-create-dense-from-template', pjoin(outdata_parpath, 'template.dscalar.nii'), pjoin(outdata_parpath, 'texture_score_nopc_n5.dscalar.nii'), '-metric', 'CORTEX_LEFT', pjoin(outdata_parpath, 'texture_score_left_nopc_n5.func.gii'), '-metric', 'CORTEX_RIGHT', pjoin(outdata_parpath, 'texture_score_right_nopc_n5.func.gii')]
    subprocess.call(' '.join(merge_command_list_orig), shell=True)

def rm_images():
    """
    """
    rm_command_list = ['rm', pjoin(outdata_parpath, '*_lh*'), pjoin(outdata_parpath, '*_rh*'), pjoin(outdata_parpath, '*left*'), pjoin(outdata_parpath, '*right*')]
    subprocess.call(' '.join(rm_command_list), shell=True)

if __name__ == '__main__':
    subjid_all = ['sub-S'+str(i).zfill(2) for i in np.arange(1,11,1)]
    for subjid in subjid_all:
         cifti_separate(subjid)
    searchlight(subjid_all)
    cifti_merge()
    rm_images()


