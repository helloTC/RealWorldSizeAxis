from psychopy import visual, event, gui
import os
import numpy as np
import random
import pickle
from collections import Counter

pjoin = os.path.join

# Prepare window
win = visual.Window(
    size=[1000,500],
    units='pix',
    fullscr=False,
    color=[1,1,1]
    )

# Prepare subject info
expInfo = {}
expInfo['被试'] = ''
expInfo['试次'] = ''
dlg = gui.DlgFromDict(expInfo, title='Size Comparison')

subjname = dlg.data[0]
subjsess = dlg.data[1]

if os.path.isfile(subjname+'_'+subjsess+'.pkl'):
    with open(subjname+'_'+subjsess+'.pkl', 'rb') as f:
        pickle_data = pickle.load(f)
    key_record = pickle_data['key_record']
    imgidx_all = pickle_data['img_idx']
    imgidx = imgidx_all[len(key_record):]
    imgnames = pickle_data['imgnames']
    imgnum = len(imgnames)
else:
    key_record = []
    imgnames = np.array(os.listdir('images'))
    imgnum = len(imgnames)
    imgidx_all = [(i,j) for i in range(imgnum) for j in range(i+1, imgnum)]
    imgidx = imgidx_all
    random.shuffle(imgidx)
  
# Start experiment
for i in range(len(imgidx)):
    # Relax screen
    if (i+1)%500 == 0:
        msg = visual.TextStim(
        win=win, 
        text='现在可以休息一会，\n按任意继续。', 
        color=[-1,-1,-1], 
        bold=True,
        font='Arial',
        height=50)
        
        msg.draw()
        win.flip()
        rest_keys = event.waitKeys()

# Load image
    img1 = visual.ImageStim(
        win=win,
        size=(224,224),
        image=pjoin('images', imgnames[imgidx[i][0]]),
        units='pix',
        pos=(-250,0)
        )            
    img2 = visual.ImageStim(
        win=win,
        size=(224,224),
        image=pjoin('images', imgnames[imgidx[i][1]]),
        units='pix',
        pos=(250,0)
        )
    text1 = visual.TextStim(
        win=win,
        text=imgnames[imgidx[i][0]].split('.')[0],
        color=[-1,-1,-1],
        bold=True,
        pos=(-250,180),
        height=30,
        font='Arial')
    text2 = visual.TextStim(
        win=win,
        text=imgnames[imgidx[i][1]].split('.')[0],
        color=[-1,-1,-1],
        bold=True,
        pos=(250,180),
        height=30,
        font='Arial')
        
    img1.draw()
    img2.draw()
    text1.draw()
    text2.draw()
    win.flip()

# record keys
    GetKeys = event.waitKeys()
    if GetKeys[0] in ['f', 'j', 'b']:
        key_record.append(GetKeys[0])
    elif GetKeys[0] in ['q']:
        win.close()
        break
    else:
        key_record.append(None)         
    event.clearEvents()    
win.close()

# Save data
outdata = {}
outdata['subject'] = subjname
outdata['session'] = subjsess
outdata['key_record'] = key_record
outdata['img_idx'] = imgidx_all
outdata['imgnames'] = imgnames
with open(subjname+'_'+subjsess+'.pkl', 'wb') as f:
    pickle.dump(outdata, f)