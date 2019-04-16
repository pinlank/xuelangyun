
# coding: utf-8

# In[1]:


import os
from tqdm import *
import seaborn as sns
import pandas as pd
import pickle
from collections import defaultdict
import numpy as np


# In[2]:


def get_path(path):
    file_name = []
    for i in os.listdir(path):
        c = i.split('_')[0]
        file_name.append(c)
    file_name = list(set(file_name))
    return file_name


# In[3]:


def get_data(file_name,file_path):
    fe = defaultdict(list)
    for i in tqdm(file_name):
        data_b = pd.read_csv(file_path + i + '_B.csv')
        data_f = pd.read_csv(file_path + i +  '_F.csv')
        fe['idx'].append(i)
        
        fe['ai1_max_b'].append(data_b.ai1.max())
        fe['ai1_min_b'].append(data_b.ai1.min())
        fe['ai1_mean_b'].append(data_b.ai1.mean())
        
        fe['ai2_max_b'].append(data_b.ai2.max())
        fe['ai2_min_b'].append(data_b.ai2.min())
        fe['ai2_mean_b'].append(data_b.ai2.mean())
           
        fe['ai1_max_f'].append(data_f.ai1.max())
        fe['ai1_min_f'].append(data_f.ai1.min())
        fe['ai1_mean_f'].append(data_f.ai1.mean())

        
        fe['ai2_max_f'].append(data_f.ai2.max())
        fe['ai2_min_f'].append(data_f.ai2.min())
        fe['ai2_mean_f'].append(data_f.ai2.mean())


    return pd.DataFrame(fe)


# In[4]:
"""修改路径"""

train_po_path = '../data/Motor_tain/Positive/'
train_ne_path = '../data/Motor_tain/Negative/'
train_po_file = get_path(train_po_path)
train_ne_file = get_path(train_ne_path)
test_path = '../data/Motor_testP/'
test_file_name = get_path(test_path)


# In[5]:


# In[6]:


train = get_data(train_ne_file,train_ne_path)
train['result'] = 0
train_po = get_data(train_po_file,train_po_path)
train_po['result'] = 1
train = train.append(train_po).reset_index(drop=True)
train.to_csv('train.cav', index=False)

# In[ ]:


test = get_data(test_file_name,test_path,test_new_map)
test.to_csv('test.cav', index=False)


