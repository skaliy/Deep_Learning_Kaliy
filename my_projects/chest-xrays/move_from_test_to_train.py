
# coding: utf-8

# In[2]:


import shutil
import os


# In[3]:


test_folder = '/home/sathiesh/Deep_Learning_Kaliy/my_projects/chest-xrays/data/test/'
train_folder = '/home/sathiesh/Deep_Learning_Kaliy/my_projects/chest-xrays/data/train'


# In[4]:


files = os.listdir(test_folder)

for f in files:
        shutil.move(test_folder+f, train_folder)

