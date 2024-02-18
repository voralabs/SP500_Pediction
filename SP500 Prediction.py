#!/usr/bin/env python
# coding: utf-8

# In[78]:


import yfinance as yf


# In[79]:


sp500 = yf.Ticker("^GSPC")


# In[80]:


sp500 = sp500.history(period="max")


# In[81]:


sp500


# In[82]:


sp500.index


# In[83]:


sp500.plot.line(y='Close', use_index=True)


# In[84]:


del sp500["Dividends"]
del sp500["Stock Splits"]


# In[85]:


sp500


# In[86]:


sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500


# In[87]:


sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
sp500


# In[88]:


sp500 = sp500.loc["1990-01-01":].copy()
sp500


# In[89]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])


# In[90]:


from sklearn.metrics import precision_score

preds = model.predict(test[predictors])


# In[91]:


import pandas as pd

preds = pd.Series(preds, index=test.index)


# In[92]:


precision_score(test["Target"], preds)


# In[93]:


combined = pd.concat([test["Target"], preds], axis=1)
combined.plot()


# In[94]:


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


# In[95]:


def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)


# In[96]:


predictions = backtest(sp500, model, predictors)


# In[97]:


predictions["Predictions"].value_counts()


# In[98]:


precision_score(predictions["Target"], predictions["Predictions"])


# In[99]:


predictions["Target"].value_counts()/predictions.shape[0]


# In[101]:


horizons = [2,5,60,250,1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    
    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
    
    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
    
    new_predictors += [ratio_column, trend_column]


# In[104]:


sp500 = sp500.dropna()


# In[105]:


model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)


# In[108]:


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


# In[109]:


predictions = backtest(sp500, model, new_predictors)


# In[110]:


predictions["Predictions"].value_counts()


# In[111]:


precision_score(predictions["Target"], predictions["Predictions"])


# In[ ]:




