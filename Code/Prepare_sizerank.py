import os
import pandas as pd
import numpy as np

pjoin = os.path.join
parpath = '/nfs/a1/userhome/huangtaicheng/workingdir/'

real_size_pd = pd.read_csv(pjoin(parpath, 'data', 'PhysicalSize', 'SizeRank.csv'))
real_size_pd = real_size_pd.sort_values('DiagSize')

real_size_pd_out = {}
real_size_name = []
real_size_size = []
real_size_rank = []

rankpart = {1: [1,5], 2:[5,10], 3:[10,50], 4:[50,100], 5:[100,500], 6:[500,1000], 7:[1000,5000], 8:[5000, 50000]}


# for i in range(25):
#     real_size_pd_part = real_size_pd[i*4:(i+1)*4]
#     real_size_name.extend(real_size_pd_part['Name'])
#     real_size_size.extend(real_size_pd_part['DiagSize'])
#     real_size_rank.extend([i+1]*len(real_size_pd_part))

for i in rankpart.keys():
    real_size_pd_part = real_size_pd[(real_size_pd['DiagSize']>=rankpart[i][0])&(real_size_pd['DiagSize']<rankpart[i][1])]
    real_size_name.extend(real_size_pd_part['Name'])
    real_size_size.extend(real_size_pd_part['DiagSize'])
    real_size_rank.extend([i]*len(real_size_pd_part))

real_size_pd_out['name'] = real_size_name
real_size_pd_out['diag_size'] = real_size_size
real_size_pd_out['real_sizerank'] = real_size_rank

real_size_pd_out = pd.DataFrame(real_size_pd_out)

