#%%

import torch
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as io

# required for solving issue with spyder
io.renderers.default='browser'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if device =='cuda:0':
    print(torch.cuda.get_device_properties(device))

#%%

r_d = torch.linspace(start=0, end=13, steps=1122, dtype=torch.double).to(device)
r_p = torch.linspace(start=0, end=13, steps=2323, dtype=torch.double).to(device)
dir_r_d = r_d[1:] - r_d[:-1]
dir_r_p = r_p[1:] - r_p[:-1]
z_d = 1
z_p = 2
vol1 = dir_r_d*z_d
vol2 = dir_r_p*z_p
switch = 1

if max(r_d) < max(r_p) or switch == 0:
    r_a = r_d
    r_b = r_p
    vol_com = vol2
    dir_r = dir_r_p
    len2 = len(r_p)
else:
    r_a = r_p
    r_b = r_d
    vol_com = vol1
    dir_r = dir_r_d
    len2 = len(r_d)
    
j = 0
add2 = 0
mul = 0
vol3 = torch.empty(0, dtype=float)

for  el in r_a[1:]:
    add1 = 0
    
    if j+1 < len2:
        while r_b[j+1] <= el:          
            add1 += vol_com[j]
            add1 += add2
            add2 = 0
            j += 1
            if j+1 >= len2: 
                break
            
        if j+1 < len2:          
            if add1 == 0:
                mul = (r_b[j+1] - el)/dir_r[j]
                add1 = vol_com[j]*(1 - mul) + add2
                add2 += -add1
            
            elif r_b[j] != el:               
                mul = (r_b[j+1] - el)/dir_r[j]
                add1 += vol_com[j]*(1 - mul) +add2
                add2 += -vol_com[j]*(1 - mul)
                
    vol3 = torch.cat((vol3, torch.tensor([add1]) ))

if switch == 0:
    
    if j+1 < len2:
        add1 = add2  
        for i in range(j, len2-1):          
            add1 += vol2[i]             
        vol3[-1] += add1
        
    res = vol1 - vol3
      
elif len2 == len(r_p):
    res = vol1 - vol3
else:
    res = vol3 - vol2

print(torch.sum(res))
#print(res)

#%%
r_d = torch.linspace(start=0, end=13, steps=1122, dtype=torch.double).to(device)
r_p = torch.linspace(start=0, end=13, steps=2323, dtype=torch.double).to(device)
dir_r_d = r_d[1:] - r_d[:-1]
dir_r_p = r_p[1:] - r_p[:-1]
z_d = 1
z_p = 2
vol1 = dir_r_d*z_d
vol2 = dir_r_p*z_p
vol_type = 1
len1 = 0

if max(r_d) < max(r_p) or vol_type == 0:
    r_a = r_d[len1:]
    r_b = r_p
    vol_com = vol2
    dir_r = dir_r_p
    len2 = len(r_p)
else:
    r_a = r_p
    r_b = r_d[len1:]
    vol_com = vol1
    dir_r = dir_r_d
    len2 = len(r_d[len1:])
    
j = 0
add2 = 0
mul = 0
vol3 = torch.empty(0, dtype=float)

for  el in r_a[1:]:
    add1 = 0
    
    if j+1 < len2:
        while r_b[j+1] <= el:          
            add1 += vol_com[j]
            add1 += add2
            add2 = 0
            j += 1
            if j+1 >= len2: 
                break
            
        if j+1 < len2:          
            if add1 == 0:
                mul = (r_b[j+1] - el)/dir_r[j]
                add1 = vol_com[j]*(1 - mul) + add2
                add2 += -add1
            
            elif r_b[j] != el:               
                mul = (r_b[j+1] - el)/dir_r[j]
                add1 += vol_com[j]*(1 - mul) +add2
                add2 += -vol_com[j]*(1 - mul)
                
    vol3 = torch.cat((vol3, torch.tensor([add1]) ))

if vol_type == 0:
    
    if j+1 < len2:
        add1 = add2  
        for i in range(j, len2-1):          
            add1 += vol2[i]
              
        vol3[-1] += add1
        
    res = vol1 - vol3
      
elif len2 == len(r_p):
    res = vol1 - vol3
elif len2 == len(r_d[len1:]):
    #print(vol3[-1], len(vol1), vol2[-1])
    res = vol3 - vol2

print(torch.sum(res))
#%%

fig = px.line(x=r_d, y=z_d)
# Add Scatter plot
fig.add_scatter(x=r_p, y=z_p)
# Display the plot
fig.show()