# Extract activation values
from os.path import join as pjoin
import nibabel as nib
import numpy as np

parpath = '/home/hellotc/workdir/RealSizeDCNN/fMRI/SizePreference/pretest/nifti/derivatives/ciftify/'

mask = nib.load('/home/hellotc/workdir/test/realsize_fMRI_images/palm_test/origsize_mask_tval.dscalar.nii').get_data()
mask = mask[0,:59412]

subjid_all = ['sub-S'+str(subjid).zfill(2) for subjid in np.arange(1,11,1)]
contrasts = ['cope3', 'cope6', 'cope9']
bigorig_all = []
smallorig_all = []
for subjid in subjid_all:
    # cohen's d: t/sqrt(dof) = cope/(varcope*sqrt(dof))
    bigorig_ctr = []
    smallorig_ctr = []
    for ctr in contrasts:
        actval_origcope_path = pjoin(parpath, subjid, 'MNINonLinear', 'Results', 'task-size-odds', 'task-size-odds_hp200_s4_level2.feat', 'GrayordinatesStats', ctr+'.feat', 'cope1.dtseries.nii')
        actval_origvarcope_path = pjoin(parpath, subjid, 'MNINonLinear', 'Results', 'task-size-odds', 'task-size-odds_hp200_s4_level2.feat', 'GrayordinatesStats', ctr+'.feat', 'varcope1.dtseries.nii')
        actval_origdof_path = pjoin(parpath, subjid, 'MNINonLinear', 'Results', 'task-size-odds', 'task-size-odds_hp200_s4_level2.feat', 'GrayordinatesStats', ctr+'.feat', 'tdof_t1.dtseries.nii')

        cope_orig = nib.load(actval_origcope_path).get_data()[0,:]
        varcope_orig = nib.load(actval_origvarcope_path).get_data()[0,:]
        dof_orig = nib.load(actval_origdof_path).get_data()[0,:]

        cohensd = cope_orig/(np.sqrt(varcope_orig)*np.sqrt(dof_orig))
        # cohensd = cope_orig
        cohensd = cohensd[:59412]

        bigorig_ctr.append(cohensd[mask>0].mean())
        smallorig_ctr.append(cohensd[mask<0].mean())

    bigorig_all.append(bigorig_ctr)
    smallorig_all.append(smallorig_ctr)
bigorig_all = np.array(bigorig_all)
smallorig_all = np.array(smallorig_all)
