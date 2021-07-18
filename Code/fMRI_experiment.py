from psychopy import visual, event, gui, core
import os
import numpy as np
import time
import pickle
pjoin = os.path.join
import argparse

"""
fMRI Design for real-world size perception

Author: Taicheng Huang.
"""
def show_instruction(win, wait_keys=['s']):
    """
    Function to show instruction
    """
    msgcls = visual.TextStim(
        win=win,
        text='Experiment is going to start, please take care',
        color=[-1,-1,-1],
        bold=True,
        font='Arial',
        height=50)
    # msgcls.autoDraw = True
    msgcls.draw()
    win.flip()
    start_key = event.waitKeys(keyList=wait_keys)
    
def show_fixation(win, nframe_fix_period):
    """
    Function to show fixation block
    """
    fixcls = visual.TextStim(
        win=win,
        text='+',
        color=[-1,-1,-1],
        bold=True,
        font='Arial',
        height=80)
    for nFrame in range(nframe_fix_period):
        if 'escape' in event.getKeys():
            win.close()
        fixcls.draw()
        win.flip()

def _get_rdm_idx(imgnums=40, split_num=[20,20]):
    """
    Generate random image index to randomly select images.
    """
    assert sum(split_num) == imgnums, "Split number is not equal to the total number of images"
    imgidx = np.arange(imgnums)
    np.random.shuffle(imgidx)
    imgidx_rdm = []
    for i in range(len(split_num)):
        if i == 0:
            imgidx_rdm.append(imgidx[0:split_num[i]])
        else:
            imgidx_rdm.append(imgidx[sum(split_num[:i]):sum(split_num[:(i+1)])])
    return imgidx_rdm

def show_images(win, parpath, stimuli_folder, rdm_idx, nframe_img, nframe_fix, jittering=20):
    """
    Function to show images to subjects.

    Parameters:
    win: Psychopy window object.
    parpath: parent path for images.
    stimuli_folder: stimuli conditions. Select from one of Big/Small Orig/Sketch/Texture.
    rdm_idx: random index to show images.
    nframe_img: duration to show an image.
    nframe_fix: duration between two images.
    jittering: jittering magnitude.
    """
    imgname_all = np.sort(os.listdir(pjoin(parpath, 'Stimuli', stimuli_folder)))
    imgname_obj = imgname_all[np.array(rdm_idx)]

    notice_idx = np.random.choice(np.arange(3,imgname_obj.shape[0]-3),2)
    
    blankcls = visual.TextStim(
        win=win,
        text='+',
        color=[-1,-1,-1],
        bold=True,
        font='Arial',
        height=80) 
   
    for i, img in enumerate(imgname_obj):
        jitter_rdm = np.random.choice([1,-1], size=1)[0]

        imgcls = visual.ImageStim(
        win=win,
        image=pjoin(parpath, 'Stimuli', stimuli_folder, img),
        size=250,
        units='pix',
        pos=(0+jitter_rdm*jittering,0))                

        redboxcls = visual.Rect(
        win=win,
        lineColor='red',
        lineWidth=3,
        size=300,
        pos=(0+jitter_rdm*jittering,0))

        for nFrame in range(nframe_img):
            if i in notice_idx:
                imgcls.draw()
                redboxcls.draw()
                blankcls.draw()
            else:
                imgcls.draw()
                blankcls.draw()
            win.flip()
            if 'escape' in event.getKeys():
                win.close()

        for nFrame in range(nframe_fix):
            blankcls.draw()
            win.flip()
            if 'escape' in event.getKeys():
                win.close()

def show_imgblocks(win, parpath, run_sequence, fix_period_long=16.0, fix_period_short=0.30, stim_period_short=0.20):
    """
    win: psychopy win object. 
    parpath: parpath for stimuli images
    run_sequence: run_sequence as one of eight sequences. Select 1 to 8
    """
    condtransfer_dict = {'A': 'BigOrig', 'B': 'SmallOrig',
                         'C': 'BigSketch', 'D': 'SmallSketch',
                         'E': 'BigTexture', 'F': 'SmallTexture'}
     
    condsequence_all = [['ABC', 'DEF', 'FED', 'CBA'],
                        ['BAD', 'CFE', 'EFC', 'DAB'],
                        ['CBA', 'FED', 'DEF', 'ABC'],
                        ['ACE', 'BDF', 'FDB', 'ECA'],
                        ['BDF', 'ACE', 'ECA', 'FDB'],
                        ['CEA', 'DFB', 'BFD', 'AEC'],
                        ['BAC', 'EDF', 'FDE', 'CAB'],
                        ['BCA', 'EFD', 'DFE', 'ACB']]
    condsequence = condsequence_all[run_sequence]

    framerate = win.getActualFrameRate()
    refresh_time = 1.0/framerate

    nframe_fix_period = int(fix_period_long/refresh_time)
    nframe_fix = int(fix_period_short/refresh_time)
    nframe_stim = int(stim_period_short/refresh_time)
    imgnum = 40
    total_time_stimuli = len(condsequence[0])*(fix_period_short+stim_period_short)*imgnum

    time_start = time.time()
    show_fixation(win, nframe_fix_period)
    for bigblock in condsequence:
        time0_imgs = time.time()
        for smallblock in bigblock:
            print('Condition {}'.format(condtransfer_dict[smallblock]))
            rdm_idx = _get_rdm_idx(imgnum, [imgnum])[0]
            show_images(win, parpath, condtransfer_dict[smallblock], rdm_idx, nframe_stim, nframe_fix)
        time1_imgs = time.time()
        dur_imgs = time1_imgs - time0_imgs
        nframe_fix_adjust = int((total_time_stimuli-dur_imgs+fix_period_long)/refresh_time)
        show_fixation(win, nframe_fix_adjust)

        time1_imgfix = time.time()
        print('  Time for one block {}'.format(time1_imgfix-time0_imgs))
    time_end = time.time()
    print('Time for one run {}'.format(time_end-time_start))

def run_experiment(parpath, subjname, run_id):
    """
    Main function to run total experient.
    For each new participant, a new run sequence will be generated 
    and saved into the output pickle file.
    run_num selected from 1-8.
    """
    if not os.path.isfile(pjoin(parpath, 'RecordStimuli', subjname+'.pkl')):
        output = {}
        output['subjname'] = subjname
        output['run_ids'] = [run_id]
        runsequences = np.arange(8)
        np.random.shuffle(runsequences)
        output['run_sequence'] = runsequences
    else:
        with open(pjoin(parpath, 'RecordStimuli', subjname+'.pkl'), 'rb') as f:
            output = pickle.load(f)
        output['run_ids'].append(run_id)

    with open(pjoin(parpath, 'RecordStimuli', subjname+'.pkl'), 'wb') as f:
        pickle.dump(output, f)

    # Prepare window
    win = visual.Window([800,600],
    units='pix',
    fullscr=False,
    color=[0,0,0],
    )
    # Start stimuli
    # Instruction
    show_instruction(win)
    # Show stimuli
    show_imgblocks(win, parpath, output['run_sequence'][run_id-1])
    win.close()

def main():
    """
    """
    parpath = '/Users/huangtaicheng/workingdir/Code/Realsize/realsize_fMRI/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjname', dest='subjname', help='subject name', type=str)
    parser.add_argument('--runid', dest='runid', help='run ID', type=int)
    args = parser.parse_args()
    run_experiment(parpath, args.subjname, args.runid)

if __name__ == '__main__':
    # parpath = 'E:/code/realsize_fMRI/'
    # run_experiment(parpath, subjname='test', run_id=3)
    main()
