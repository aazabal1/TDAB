#!/usr/bin/env python
# coding: utf-8

# # Explore Raw Input Data

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[26]:


# Load data

data = pd.read_csv("../test_data.csv")
data


# In[27]:


# Describe data
data.describe()


# In[28]:


# Find data types
data.dtypes


# In[30]:


# Check Null and NaN values

sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')#.get_figure().savefig("missing.png")


# There are only a few Null values, so the best thing would be to keep them, as this is a time series

# In[31]:


# Find where nulls are
data[data.isnull().any(axis=1)]


# In[45]:


# Better to fillna with previous value, rather than drop, as this is a time series

#data = data.dropna()
data = data.fillna(method='ffill')
data = data.drop_duplicates() # Drop duplicates
data


# In[33]:


# Plot ModePilote distribution
try:
    sns.displot(data["ModePilote"])
except RuntimeError as re:
    if str(re).startswith("Selected KDE bandwidth is 0. Cannot estimate density."):
        sns.displot(data["ModePilote"], kde_kws={'bw': 0.1})


# This confirms that the column "ModePilote" is a discrete column containing 2 or 5

# In[34]:


# Plot distribution of all features to understand what's going on

fig, axes = plt.subplots(nrows=9, ncols=3, figsize=(15,30))

for i, column in enumerate(data.columns):
    if not column=="DateTime":
        try:
            sns.distplot(data[column],ax=axes[i//3,i%3])
        except RuntimeError as re:
            if str(re).startswith("Selected KDE bandwidth is 0. Cannot estimate density."):
                sns.distplot(data[column], kde_kws={'bw': 0.1},ax=axes[i//3,i%3])
        


# Some features seem to have a huge peak, probably due to tacking. Plot this to confirm
# 

# In[35]:


sns.displot(data, x='SoS', hue='Tacking',kde=True)


# In[36]:


sns.displot(data, x='Roll', hue='Tacking',kde=True)


# In[37]:


sns.displot(data, x='Yaw', hue='Tacking',kde=True)


# In[38]:


sns.displot(data, x='ModePilote', hue='Tacking',kde=True)


# In[39]:


data["Yaw"].describe()


# In[40]:


# Yaw is an angle that can only go from 0 to 360, not -360 to 360, so apply a transformation

def abs_angle(angle):
    if angle<0:
        abs_angle = angle+360
    else:
        abs_angle = angle
    return abs_angle


# In[41]:


data["Abs_Yaw"] = data.apply(lambda row: abs_angle(row['Yaw']),axis=1)


# In[42]:


data["Abs_Yaw"].describe()


# In[43]:


sns.displot(data, x='Abs_Yaw', hue='Tacking',kde=True)


# In[13]:


import plotly as py
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
pio.renderers.default = "notebook"

from itertools import cycle


# In[55]:


print(py.__version__)


# In[44]:


# Apply rolling mean to smooth data
data_rw = data.rolling(60).mean()
data_rw


# Plot time series of some of the features of interest to understand behaviour

# In[47]:


transparency = 0.4
is_scaled = None

pred_fig = go.Figure()
color_cycle = cycle(px.colors.qualitative.Set1)

color = next(color_cycle)
line_style = {"color":color, "dash":"solid","width":1.5}
pred_fig.add_trace(go.Scatter(
        x=data["DateTime"],
        y=data["CurrentSpeed"],
        mode="lines", line=line_style))

color = next(color_cycle)
line_style = {"color":color, "dash":"solid","width":1.5}
pred_fig.add_trace(go.Scatter(
        x=data["DateTime"],
        y=data["CurrentSpeed"].rolling(60).mean(),#/data_rw["Longitude"].rolling(60).mean().max(),
        mode="lines", line=line_style))

color = next(color_cycle)
line_style = {"color":color, "dash":"solid","width":1.5}
pred_fig.add_trace(go.Scatter(
        x=data["DateTime"],
        y=data["Tacking"],
        mode="lines", line=line_style))

color = next(color_cycle)
line_style = {"color":color, "dash":"solid","width":1.5}
pred_fig.add_trace(go.Scatter(
        x=data["DateTime"],
        y=data["Latitude"]/data["Latitude"].max(),
        mode="lines", line=line_style))

color = next(color_cycle)
line_style = {"color":color, "dash":"solid","width":1.5}
pred_fig.add_trace(go.Scatter(
        x=data["DateTime"],
        y=data["Longitude"]/data["Longitude"].max(),
        mode="lines", line=line_style))

color = next(color_cycle)
color = next(color_cycle)
line_style = {"color":color, "dash":"solid","width":1.5}
pred_fig.add_trace(go.Scatter(
        x=data["DateTime"],
        y=data["ModePilote"],
        mode="lines", line=line_style))


# Plot boat trajectory (latitude vs longitude)

# In[52]:


transparency = 0.4
is_scaled = None

pred_fig = go.Figure()
color_cycle = cycle(px.colors.qualitative.Set1)

color = next(color_cycle)
line_style = {"color":color, "dash":"solid","width":1.5}
pred_fig.add_trace(go.Scatter(
        x=data["Latitude"],
        y=data["Longitude"],
        mode="lines", line=line_style))
               


# Select only data after 9:46:40 on 2019-04-14. This is the point at which the boat seems to be actually starting to move

# In[53]:


data_start = data[data["DateTime"]>= '2019-04-14 09:46:40'].reset_index(drop=True)
data_start


# In[54]:


data_start.groupby(["Tacking"]).describe()


# This shows that we are dealing with a heavily imbalanced class dataset, which will have to be accounted for

# Some of the main aspects of the dataset (data type, format, etc) have been idenfitied, as well as some important data cleaning steps that will be properly done later

# In[ ]:


data_start.to_csv("../data/cleaned_data.csv")

