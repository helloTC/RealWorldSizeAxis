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
    _cifti_separate(subject, 'cope3')
    _cifti_separate(subject, 'cope6')
    _cifti_separate(subject, 'cope9')

def searchlight(subject, n=3):
    """
    """
    print('Search light analysis for subject {}'.format(subject))
    # Prepare data
    geometry_lh = nib.load(pjoin(outdata_parpath, 'L_midthickness.surf.gii')).darrays[1].data
    geometry_rh = nib.load(pjoin(outdata_parpath, 'R_midthickness.surf.gii')).darrays[1].data
    sizeorig_lh = nib.load(pjoin(outdata_parpath, subject+'_'+'cope3'+'_left.func.gii')).darrays[0].data
    sizeorig_rh = nib.load(pjoin(outdata_parpath, subject+'_'+'cope3'+'_right.func.gii')).darrays[0].data
    sizesketch_lh = nib.load(pjoin(outdata_parpath, subject+'_'+'cope6'+'_left.func.gii')).darrays[0].data
    sizesketch_rh = nib.load(pjoin(outdata_parpath, subject+'_'+'cope6'+'_right.func.gii')).darrays[0].data
    sizetexture_lh = nib.load(pjoin(outdata_parpath, subject+'_'+'cope9'+'_left.func.gii')).darrays[0].data
    sizetexture_rh = nib.load(pjoin(outdata_parpath, subject+'_'+'cope9'+'_right.func.gii')).darrays[0].data
    # Extract tstats
    corr_origsketch_lh = np.zeros((32492))
    corr_origsketch_rh = np.zeros((32492))
    corr_origtexture_lh = np.zeros((32492))
    corr_origtexture_rh = np.zeros((32492))
    for i in range(32492):
        if (i+1)%1000 == 0:
            print('  Vertex {}'.format(i+1))
        neighbor_vertex_lh = surf_tools.get_n_ring_neighbor([i], geometry_lh, n=3)[0]
        neighbor_vertex_lh = np.array(list(neighbor_vertex_lh))
        neighbor_vertex_rh = surf_tools.get_n_ring_neighbor([i], geometry_rh, n=3)[0]
        neighbor_vertex_rh = np.array(list(neighbor_vertex_rh))

        sizeorig_lh_tmp = sizeorig_lh[neighbor_vertex_lh]
        sizeorig_rh_tmp = sizeorig_rh[neighbor_vertex_rh]
        sizesketch_lh_tmp = sizesketch_lh[neighbor_vertex_lh]
        sizesketch_rh_tmp = sizesketch_rh[neighbor_vertex_rh]
        sizetexture_lh_tmp = sizetexture_lh[neighbor_vertex_lh]
        sizetexture_rh_tmp = sizetexture_rh[neighbor_vertex_rh]

        if (0 in sizeorig_lh_tmp) or (0 in sizesketch_lh_tmp) or (0 in sizetexture_lh_tmp):
            continue
        corr_origsketch_lh[i] = stats.pearsonr(sizeorig_lh_tmp, sizesketch_lh_tmp)[0]
        corr_origsketch_rh[i] = stats.pearsonr(sizeorig_rh_tmp, sizesketch_rh_tmp)[0]
        corr_origtexture_lh[i] = stats.pearsonr(sizeorig_lh_tmp, sizetexture_lh_tmp)[0]
        corr_origtexture_rh[i] = stats.pearsonr(sizeorig_lh_tmp, sizetexture_rh_tmp)[0]
    iogii_origsketch_lh = iofiles.make_ioinstance(pjoin(outdata_parpath, 'corr_origsketch_'+subject+'_lh.func.gii'))
    iogii_origsketch_rh = iofiles.make_ioinstance(pjoin(outdata_parpath, 'corr_origsketch_'+subject+'_rh.func.gii'))
    iogii_origtexture_lh = iofiles.make_ioinstance(pjoin(outdata_parpath, 'corr_origtexture_'+subject+'_lh.func.gii'))
    iogii_origtexture_rh = iofiles.make_ioinstance(pjoin(outdata_parpath, 'corr_origtexture_'+subject+'_rh.func.gii'))
    iogii_origsketch_lh.save(corr_origsketch_lh, 'CortexLeft')
    iogii_origsketch_rh.save(corr_origsketch_rh, 'CortexRight')
    iogii_origtexture_lh.save(corr_origtexture_lh, 'CortexLeft')
    iogii_origtexture_rh.save(corr_origtexture_rh, 'CortexRight')
    
def cifti_merge(subject):
    """
    """
    print('Merge into CIFTI for subject {}'.format(subject))
    merge_command_list_origsketch = ['wb_command', '-cifti-create-dense-from-template', pjoin(outdata_parpath, 'template.dtseries.nii'), pjoin(outdata_parpath, 'corr_origsketch_'+subject+'.dscalar.nii'), '-metric', 'CORTEX_LEFT', pjoin(outdata_parpath, 'corr_origsketch_'+subject+'_lh.func.gii'), '-metric', 'CORTEX_RIGHT', pjoin(outdata_parpath, 'corr_origsketch_'+subject+'_rh.func.gii')]
    merge_command_list_origtexture = ['wb_command', '-cifti-create-dense-from-template', pjoin(outdata_parpath, 'template.dtseries.nii'), pjoin(outdata_parpath, 'corr_origtexture_'+subject+'.dscalar.nii'), '-metric', 'CORTEX_LEFT', pjoin(outdata_parpath, 'corr_origtexture_'+subject+'_lh.func.gii'), '-metric', 'CORTEX_RIGHT', pjoin(outdata_parpath, 'corr_origtexture_'+subject+'_rh.func.gii')]
    subprocess.call(' '.join(merge_command_list_origsketch), shell=True)
    subprocess.call(' '.join(merge_command_list_origtexture), shell=True)

def rm_images():
    """
    """
    rm_command_list = ['rm', pjoin(outdata_parpath, '*_lh*'), pjoin(outdata_parpath, '*_rh*'), pjoin(outdata_parpath, '*left*'), pjoin(outdata_parpath, '*right*')]
    subprocess.call(' '.join(rm_command_list), shell=True)

if __name__ == '__main__':
    for subj in ['sub-S'+str(i).zfill(2) for i in np.arange(2,11,1)]:
        cifti_separate(subj)
        searchlight(subj)
        cifti_merge(subj)
    rm_images()


