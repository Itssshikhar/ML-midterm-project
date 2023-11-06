#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests


# In[2]:


url = "http://localhost:6969/model_predict"


# In[3]:


stock = {
    "symbol": "AOS",
    "name": "A.O. Smith Corp",
    "sector": "Industrials",
    "price_earnings": 27.76,
    "dividend_yield": 1.1479592,
    "earnings_share": 1.7,
    "52_week_low": 68.39,
    "52_week_high": 48.925,
    "market_cap": 10783419933,
}


# In[4]:


stock


# In[6]:


response = requests.post(url, json=stock).json()
print(response)

# In[ ]:




