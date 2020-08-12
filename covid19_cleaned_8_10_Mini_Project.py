#!/usr/bin/env python
# coding: utf-8

# In[148]:


#covid19_cleaned_8_10.csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[149]:


DF=pd.read_csv("covid19_cleaned_8_10.csv",low_memory = False)
y=np.array(DF["deceased"]) 
x=np.array(DF[["age",'chronic_disease_binary','travel_history_binary']])


# In[150]:


len(x)# make sure lel(x) and len(y) must be the same 


# In[151]:


len(y)


# In[196]:


DF.columns


# In[197]:


DF['deceased']


# In[199]:


DF.age.tolist()


# In[ ]:





# In[ ]:





# In[153]:


# logistic looking at age =x we will predict y= dead or hospitalised 
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[154]:


y=y.ravel() #we are expecting y.shape to be (,2309996)


# In[155]:


y.shape


# In[156]:


#x=x.reshape(-1,1)


# In[157]:


# Make your x is my age=x two dementional
#x=x.reshape(-1,1) 
#y=y.ravel() # my y was (n,2) I had to make it into 1D
#y.shapey.shape
x.shape


# In[158]:


# make sure it is ( 2309996,1)


# In[159]:


model=LR() # Logistic Regression 
xtrain,xtest,ytrain,ytest=train_test_split(x,y)
model.fit(xtrain,ytrain) # Doing LOgistic Model my training data 
yp=model.predict(xtest) # tell the prediction, using xtest 
accuracy_score(ytest,yp) # ytest real values of xtest, yp is model using train
# Below we can see what is the percenttage of accuracy of predicting looking at x (age) will be y (deseased) 


# In[160]:


ACCS=[] # how important is the randomization?
for j in range(25):
    model=LR()
    xtrain,xtest,ytrain,ytest=train_test_split(x,y)
    model.fit(xtrain,ytrain)
    YP=model.predict(xtest)
    ACCS.append(accuracy_score(ytest,YP)) # make a list of all 25 running model


# In[161]:


plt.boxplot(ACCS,vert=False);
plt.title("Accuracy for AGE") # Looking at age we can tell what is the percentage of the person taht going to be desseased  


# In[162]:


from sklearn.model_selection import cross_validate


# In[163]:


cross_validate(LR(), x, y, cv=5)


# In[164]:


# Logistic model, 
mymodel=LR()
xtrain,xtest,ytrain,ytest=train_test_split(x,y)# split the data into train and test 
mymodel.fit(xtrain,ytrain)
ypredict=mymodel.predict(xtest)


# In[165]:


probabilities=mymodel.predict_proba(xtest) [:,1]# all rows and second columns 


# In[166]:


# import the ROC curve Package 
from sklearn.metrics import roc_curve


# In[167]:


FPR, TPR, THRESHOLD = roc_curve(ytest, probabilities)# False Positive Ratio, True Positive Ratio


# In[168]:


plt.plot(FPR,TPR,c="red")
plt.plot([0,1],[0,1],c="k",ls="--")
plt.grid()
plt.xlabel("False Positive Rate",fontsize=14)
plt.ylabel("True Positive Rate",fontsize=14)
plt.gcf().set_size_inches(6,6) 
# Higher  the corve is better accuracy , prediction  NOT GOOD??????


# In[169]:


fig,ax=plt.subplots(nrows=1,ncols=1)
for j in range(10):
    mymodel = LR()
    xtrain,xtest,ytrain,ytest=train_test_split(x,y)
    
    mymodel.fit(xtrain,ytrain)
    probs=mymodel.predict_proba(xtest)[:,1]
    FPR,TPR,THRESHOLDS=roc_curve(ytest,probs)
    ax.plot(FPR,TPR)
    
ax.grid()
ax.plot([0,1],[0,1],c="k",ls="--") # 45 degree dashed black line
ax.set_xlabel("False Positives")
ax.set_ylabel("True Positives")
ax.set_title("Covid-19 First 1000 Cases: Age vs Deceased 10 Train/Test Splits")
fig.set_size_inches(6,6)


# ## Leave one out cross validation 
# 

# In[170]:


# Leave One out cross valdation 
import matplotlib.pyplot as plt
import sklearn

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut

#loocv = model_selection.LeaveOneOut() # Leave One out CV
#model_loocv = LogisticRegression()
#results_loocv = model_selection.cross_val_score(model_loocv, x, y, cv=loocv)
#print("Accuracy: %.2f%%" % (results_loocv.mean()*100.0))


# ## conclusion : 
# The mean accuracy for the model using the leave-one-out cross-validation is ___%  percent.

# In[171]:


DF.columns


# In[172]:


DF['sex']=DF['sex'].apply(lambda x: 0 if x =='female' else 1)


# In[173]:


DF.sex.unique()


# In[174]:


Y=DF["sex"] # my target using almost all features 
np.unique(Y,return_counts=True) # So there is 1158473 zeros and ..


# In[175]:


#DF2 = pd.get_dummies(DF2,columns=["country"])
DF["country"].isna().sum()


# In[211]:


dummies = pd.get_dummies(DF,columns=["country"])# turn country column into dummies 


# In[212]:


dropcols=DF[['Unnamed: 0','ID','city', 'province','latitude', 'longitude', 'date_onset_symptoms','date_admission_hospital','travel_history_dates', 'travel_history_location','reported_market_exposure','chronic_disease','outcome', 'date_death_or_discharge', 'admin3', 'admin2', 'admin1','fever', 'cough','fatigue', 'headache', 'dizziness', 'sore throat', 'pneumonia','respiratory', 'nausea', 'diarrhea', 'severe_r']]
# DO NOT FOGET TO DRROP YOUR TARGET
DF.head()


# In[213]:


#X=np.array(DF.drop(columns=dropcols)) # My x columns are AGE, CHRONIC DESEASE,COUNRTY,TRAVEL_HISTORY_BINARY,HOSPIT,DECEASED


# In[214]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier as ANN
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.model_selection import train_test_split


# In[215]:


# SCALE MY X standartize 
#scaler=MinMaxScaler()
#XScaled=scaler.fit_transform(X)
DF["travel_history_binary"].unique()


# ## NNA predict sex using "age","chronic_disease_binary","hospitalized","deceased","travel_history_binary"]],dummies],axis=1)
# dummies are country 

# In[216]:



xvals=["age","chronic_disease_binary","hospitalized","deceased","travel_history_binary"]

# X=np.array(pd.concat([DF[["age","chronic_disease_binary","hospitalized","deceased","travel_history_binary"]],
                      #dummies],axis=1))
xscal=DF[xvals]
# I want to scale first all those numerical ones then concatinate with dummies 
#scaler=MinMaxScaler()
#XScaled=scaler.fit_transform(xscal)
#X=pd.concat([XScaled,dummies],axis=1)


# In[217]:


y=np.array(DF["sex"]) 


# In[ ]:


X=np.array(pd.concat([DF[["age","chronic_disease_binary","hospitalized","deceased","travel_history_binary"]],
                      dummies],axis=1))


# In[221]:


from sklearn.neural_network import MLPClassifier as ANN
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.model_selection import train_test_split


# In[ ]:


#model=ANN(hidden_layer_sizes=(3,4,4), max_iter=10)
#acc=[]
#for repeat in range(10):
    #xtrain,xtest,ytrain,ytest=train_test_split(X, Y) # I am using x standardized 
    #model.fit(xtrain,ytrain)
    #YP=model.predict(xtest)
    #print(accuracy_score(ytest,YP))
    #acc.append(accuracy_score(ytest,YP))
    


# In[ ]:


#plt.boxplot(acc, vert=False)
#plt.grid()
#plt.gcf().set_size_inches(10,2) 

