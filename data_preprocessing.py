
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
get_ipython().magic(u'matplotlib inline')


#                                      1. Data exploration and sampling

# In[2]:

data = pd.read_csv('assignment_data.csv')
pd.options.display.max_columns = None


# The ad click result is apparently unbalanced. Considering the large size of the data, down sampling the 'not click' type is quick and feasible 

# In[3]:

data['ad_click'].hist()


# In[4]:

# Down sampling, make each type 50-50
n_click = len(data.loc[data['ad_click'] == 1,'ad_click'])
click_indices = np.array(data.loc[data['ad_click'] == 1,'ad_click'].index)
noclick_indices = len(data.loc[data['ad_click'] == 0,'ad_click'].index)
rd_noclick_indices = np.random.choice(noclick_indices,n_click,replace = False)  #randomly pick n number from '0' class
undersample_indices = np.concatenate([click_indices,rd_noclick_indices])
u_data = data.ix[undersample_indices,:]
u_data = pd.DataFrame(u_data,columns = list(u_data))


#                                 2  Preprocessing and feature cleaning: 
# <br>2.1 Rewrite user and dest location_navigation, keep the name of continent, country and next level of country-- call     it 'region'

# In[5]:

def reformat_navigation(df,user):
    new_loc = df[user + '_location_navigation_string'].str.split('|',expand = True)
    cont_loc = new_loc.ix[:,1].str.split(':',expand = True).iloc[:,2]
    coun_loc = new_loc.ix[:,2].str.split(':',expand = True).iloc[:,2]
    region_loc = new_loc.ix[:,3].str.split(':',expand = True).iloc[:,2]
    df[user +'_loc_continent'] = cont_loc
    df[user + '_loc_country'] = coun_loc
    df[user + '_loc_region'] = region_loc
    
reformat_navigation(u_data,'user')
reformat_navigation(u_data,'dest')


# 2.2 Add a new variable user-desitnation distance, calculated by latitude and longtitude of 2 places

# In[6]:

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

u_data['dist'] = haversine_np(u_data['user_longitude'],u_data['user_latitude'],u_data['dest_longitude'], u_data['dest_latitude'])


# 2.3 Generate new features base on search_timestamp, check_in_date, check_out_date

# In[7]:

u_data['search_timestamp'] = pd.to_datetime(u_data['search_timestamp'])
u_data['check_in_date'] = pd.to_datetime(u_data['check_in_date'], format='%Y-%m-%d', errors="coerce")
u_data['check_out_date'] = pd.to_datetime(u_data['check_out_date'], format='%Y-%m-%d', errors="coerce")
props = {}
for prop in ['month', 'day', 'hour', 'minute', 'dayofweek', 'quarter']:
    props[prop] = getattr(u_data['search_timestamp'].dt, prop)
    
carryover = [p for p in u_data.columns if p not in ['search_timestamp', 'check_in_date', 'check_out_date']]
for prop in carryover:
    props[prop] = u_data[prop]
    
date_props = ["month", "day", "dayofweek", "quarter"]
for prop in date_props:
    props['ci_{0}'.format(prop)] = getattr(u_data['check_in_date'].dt, prop)
    props['co_{0}'.format(prop)] = getattr(u_data['check_out_date'].dt, prop)
props['stay_span'] = (u_data['check_out_date'] - u_data['check_in_date']).astype('timedelta64[h]')
        
ret_data = pd.DataFrame(props)


# 2.4 Remove non-numeric columns user_id, session_id, search_id, search_result_id, ad_id

# In[8]:

ret_data = ret_data.drop(['user_id','session_id','user_location_navigation_string','advertiser_id',
                          'search_id','dest_location_navigation_string','search_result_id','search_result_timestamp','ad_id','ad_click_timestamp'],1)


# 2.5 Dealing with categorical feature using LabelEcoder

# In[9]:

num_data = ret_data.drop(['ua_browser_name','ua_device_category','ua_device_os',
                  'site_language','user_loc_continent','user_loc_country','user_loc_region',
                  'dest_loc_continent','dest_loc_country','dest_loc_region'],1)
cat_data = ret_data[['ua_browser_name','ua_device_category','ua_device_os',
                  'site_language','user_loc_continent','user_loc_country','user_loc_region',
                    'dest_loc_continent','dest_loc_country','dest_loc_region']]
d = defaultdict(LabelEncoder)
cat_data = cat_data.apply(LabelEncoder().fit_transform)
cat_data.shape


# In[10]:

num_data.shape


#       Now we have 7 categorical and 32 numerical features, and all categorical features are encoded

# In[11]:

my_data = pd.concat([num_data,cat_data],axis = 1)
my_data.head(10)


# 1.3 Finding correlations:
#     no significant correlation between features except some demographic features such as between latitude and longtitude bwtween users and destinations

# In[12]:

corrr = my_data.corr(method='pearson', min_periods=1)
fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(corrr,linewidths=.5, fmt="d")


# Take a quick look at variables correlation, there is no direct linear correlations between ad click and other variables. Except for some within-location and within-time relation, there are a few interesting findings:
# <br> <b> 1. Mysterious_feature_1 is strongly indcating some feature of destination --- it's correlated with destination_longtitude; but also possible user location
# <br> <b> 2. Mysterious_feature_2 is somewhat season/'time of a year' related  --- it's correlated with check out month and quater

# 1.4 Missing value imputation and write out preprocessed data

# In[13]:

my_data.to_csv('my_data.csv',index = False)


# In[ ]:



