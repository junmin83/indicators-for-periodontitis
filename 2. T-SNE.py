#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
import sklearn
from sklearn.manifold import TSNE
os.getcwd()


# In[2]:


# Call data
features =np.load("features.npy")
label =np.load("label.npy")
label = label.reshape(-1,)


# ## TSNE

# In[3]:


model = TSNE(learning_rate=50, random_state=70)
transformed = model.fit_transform(features)

xs = transformed[:,0]
ys = transformed[:,1]

SP_xs = np.array([])
SP_ys= np.array([])

NSP_xs = np.array([])
NSP_ys= np.array([])

neg_xs= np.array([])
neg_ys= np.array([])


for i in range(label.shape[0]):
    if label[i] == 2:
        SP_xs = np.append(SP_xs,xs[i])
        SP_ys = np.append(SP_ys,ys[i])
    if label[i] == 1:
        NSP_xs = np.append(NSP_xs,xs[i])
        NSP_ys = np.append(NSP_ys,ys[i])
    if label[i] == 0:
        neg_xs = np.append(neg_xs,xs[i])
        neg_ys = np.append(neg_ys,ys[i])



plt.scatter(NSP_xs,NSP_ys,c='green',s=0.3,marker="X", alpha=0.6, label='NSP')
plt.scatter(SP_xs,SP_ys,c='red',s=0.3, alpha=0.8, label='SP')
plt.legend()
plt.axis('off')
# plt.savefig('TSNE-NSPvsSP.jpg', dpi=1000, bbox_inches='tight')
plt.show()



pos_xs = np.array([])
pos_ys= np.array([])

for i in range(label.shape[0]):
    if label[i] == 2:
        pos_xs = np.append(pos_xs,xs[i])
        pos_ys = np.append(pos_ys,ys[i])
    if label[i] == 1:
        pos_xs = np.append(pos_xs,xs[i])
        pos_ys = np.append(pos_ys,ys[i])

plt.scatter(neg_xs,neg_ys,c='blue',s=0.3,alpha=0.9,label='NoP')
plt.scatter(pos_xs,pos_ys,c='red',marker="X",s=0.3, alpha=0.4, label='TP')
plt.legend()
plt.axis('off')
# plt.savefig('TSNE-NoPvsTP.jpg', dpi=1000, bbox_inches='tight')
plt.show()
    


# In[ ]:





# In[ ]:





# In[ ]:




