
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import sys
sys.path.append("../../fastai/")


# In[ ]:


from fastai.imports import *
from fastai.plots import *


# In[ ]:


PATH = "/home/sathiesh/Deep_Learning_Kaliy/my_projects/chest-xrays/data/"  #MMIV
#PATH = "/Users/Sathiesh/MachineLearning/Deep_Learning_Kaliy/my_projects/chest-xrays/data/" #Home


# In[ ]:


label_csv = f'{PATH}Data_Entry_2017.csv' 


# In[ ]:


df = pd.read_csv(label_csv).drop(['Follow-up #','Patient ID', 'Patient Age', 'Patient Gender', 'View Position',
              'OriginalImageWidth', 'OriginalHeight', 'OriginalImagePixelSpacingx', 'OriginalImagePixelSpacingy', 'Unnamed: 11'],axis=1)


# In[ ]:


df_sorted_diseases = df.pivot_table(index='Finding Labels', aggfunc=len).sort_values('Image Index', ascending=False) 


# In[ ]:


def chestX_info(): 
    df_sorted_diseases[0:10].plot(kind='bar',figsize=(13,8))


# In[ ]:


def disease_finding(disease_name):
    df_disease = df_sorted_diseases[df_sorted_diseases.index.str.contains(disease_name)]
    disease_row = df_sorted_diseases.loc[disease_name]['Image Index']
    total_apperance = sum(df_disease['Image Index'])
    print(disease_name +': ' + str(disease_row) + '\n'+ 'Total apperance: ' + str(total_apperance)+ '\n'
      +'Number of rows that must be deleted: ' + (str)(total_apperance - disease_row))


# In[ ]:


def remove_comined_rows(disease_name):
    temp_df = df
    temp_df['searching_disease'] = temp_df['Finding Labels'].str.contains(disease_name)
    temp_df = temp_df[temp_df['searching_disease']]; #all rows with disease_name 

    temp_df['Finding Labels'] = temp_df['Finding Labels'].str.replace('|','REMOVE')
    temp_df['searching_disease'] = temp_df['Finding Labels'].str.contains('REMOVE')
    arr = np.array(temp_df[temp_df['searching_disease']].index) #labeled as disease_name + another disease
    return arr;


# In[ ]:


def clean_label(disease_name):
    disease_df = df
    
    disease_df = disease_df.drop(remove_comined_rows(disease_name))
    disease_df = disease_df.reset_index(drop=True)
    
    df_sorted = disease_df.pivot_table(index='searching_disease', aggfunc=len).sort_values('Image Index') 
    df_sorted.index = [disease_name,'Other']
    
    disease_count = df_sorted['Image Index'][0]
    other_disease_count = df_sorted['Image Index'][1]
    ratio = (other_disease_count// disease_count)-1
    disease_df = disease_df.append([disease_df[disease_df.searching_disease]]*ratio, ignore_index=True)
    
    df_sorted = disease_df.pivot_table(index='searching_disease', aggfunc=len).sort_values('Image Index') 
    df_sorted.index = [disease_name,'Other']
    df_sorted['Image Index'].plot(kind='bar', title="Other diseases vs "+disease_name)
    
    disease_df = disease_df.drop(['Finding Labels'], axis=1)
    
    cleaned_label_path = f'{PATH}/cleaned_labels/{disease_name}.csv'
    disease_df.to_csv(cleaned_label_path, index=False)
    return disease_df; 

