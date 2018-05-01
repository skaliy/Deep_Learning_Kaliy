
# coding: utf-8

# In[11]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


import os
import shutil
import pandas as pd


# In[ ]:


def move_to_test(df_test): 
    for i  in df_test['Image Index']: 
    shutil.move(f'{PATH}train/{i}', f'{PATH}test/{i}')  

