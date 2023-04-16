#!/usr/bin/env python
# coding: utf-8

# ### Section 2: ML Practical

# #### Part 1: Loading all the Python libraries for the ML Model 

# In[ ]:


#Importing library for loading the data into Python
import pandas as pd

#Importing library for Pre-processing of the data
from sklearn import preprocessing

#Importing library for getting all the models
from sklearn.tree import DecisionTreeClassifier

#Importing library for splitting the data for learning and testing phases
from sklearn.model_selection import train_test_split

#Importing library for getting all the metrics for performance evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score

#Importing library for saving the models with the results
#import pickle
#import zipfile
import os
import joblib
import argparse

#Importing this Azure library to run the code in the environment...
from azureml.core import Run    
run = Run.get_context()


# #### Part 2: Data Ingestion and Preprocessing 

# In[ ]:


#Loading the dataset and print the data types of the columns...
data = pd.read_csv("data1.csv")
df = data.infer_objects()
print(df.dtypes[0:14])


# In[ ]:


# Removing the empty instances (or instances with white columns) of the dataset (lines in the file)
for i in range (0,14):
    if df.dtypes[i] != 'int64':
        data.iloc[:,i] = df.iloc[:,i].map(lambda x:x.strip())
array = data
array.head()


# In[ ]:


# Seperating the input columns and the target (output) columns...
inputs = array.drop('class', axis='columns')
target = array['class']


# In[ ]:


#Converting categorical variables into non-categorical counterparts...

labelenc = preprocessing.LabelEncoder()

X= inputs.values
Y= target.values

for i in range (0,14):
    X[:,i] = labelenc.fit_transform(X[:,i])
    
#This is how X (input array) and Y (output array) now looks like...
print(X)


# In[ ]:


print(Y)


# #### Part 3: Splitting the Dataset

# In[ ]:


# Splitting the dataset into a Validation Set...
test_ratio = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = test_ratio)


# In[ ]:


#Visualizing the shape of the Training Dataset...
print(X_train.shape)
print(Y_train.shape)


# In[ ]:


#Visualizing the shape of the Testing Dataset...
print(X_test.shape)
print(Y_test.shape)


# #### Part 4: Training and Evaluating the Model 

# In[ ]:


M1 = DecisionTreeClassifier(criterion='entropy',max_features=13,max_depth=10)
run.log("DT Model Parameters",M1.get_params())

# In[ ]:


#Fitting and Testing the Decision Tree Model (M1)...
M1 = M1.fit(X_train,Y_train)
M1_pred = M1.predict(X_test)


# In[ ]:


print("--------------------Model 1: Decision Tree--------------------------")
print("Accuracy Score: ", accuracy_score(Y_test,M1_pred)*100,"%")
print("Confusion Matrix: \n", confusion_matrix(Y_test,M1_pred))
print("Hamming Loss: ", hamming_loss(Y_test,M1_pred)*100,"%")
print("F1 Score: ", f1_score(Y_test,M1_pred))
print("Average Precision Curve: ", average_precision_score(Y_test,M1_pred))

#Printing the results in the Environment as well...
run.log("Accuracy Score:",accuracy_score(Y_test,M1_pred)*100,"%")
run.log("Hamming Loss:",hamming_loss(Y_test,M1_pred)*100,"%")
run.log("F1 Score: ",f1_score(Y_test,M1_pred))
run.log("Average Precision Curve: ",average_precision_score(Y_test,M1_pred))

# In[ ]:


#Saving the first Model into a Pickle File...
# print("Export the model to DTmodel.pkl")
# f1= open('M1.pkl','wb')
# pickle.dump(M1,f1)
# f1.close
print("Export the model to DTmodel_logs.pkl")
joblib.dump(M1, "DTmodel_logs.pkl")


# In[ ]:


#Saving the first Model into a Zip File...
#zipfile.ZipFile('model1.zip',mode='w').write('M1.pkl')

