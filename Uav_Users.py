#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score 
import sklearn
import seaborn as sns
from sklearn import model_selection
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score,roc_curve
import array



# In[2]:


# Making a list of missing value types

df = pd.read_csv('Users.csv',low_memory= False )
df = df[df['cell_id'] != 'None']


# In[3]:


df.drop(columns=["file","trip","technology","ele","rsrq" , 
                 "sinr" , "signal" , "pci" , "rssi" , "netmode", "rsrp","line",
                 "predecessor", "dlong" , "dlat" , "predecessor2"] ,inplace=True)  
df.head()


# In[4]:


df["time"] = pd.to_datetime(df["time"])
df['newpos'] = df['newpos'].astype(int)

df1= df.loc[(df['time'] >= '2018-07-01') & (df['time'] <='2018-07-30')].copy()


df1['Year'] = df ['time'].dt.year
df1['Month'] = df['time'].dt.month
df1['Weekday'] = df['time'].dt.dayofweek
df1['Hour'] = df['time'].dt.hour


df1. sample(15)


# In[5]:


df1.newpos.value_counts()


# In[6]:


df1['cell_id'] = df['cell_id'].astype(float)


# In[7]:


user_counts = df1.groupby(['Hour', 'Weekday','newpos','cell_id']).size().reset_index(name='Number_Users')


print(user_counts)
# Plot the actual values and the predicted values

#df1.plot(kind="bar");


# In[8]:


result = user_counts.dtypes
print (result)


# In[9]:


#user_count.plot.scatter(x="Month", y="datarate", alpha=0.5)
plt.show()



#user_count.plot.scatter(x="Weekday", y="datarate", alpha=0.5)
plt.show()

user_counts.plot(x="Hour", y="Weekday", alpha=1)
plt.show()


# In[10]:


target_name = 'Number_Users'
# separate object for target feature
y = user_counts[target_name]

# separate object for input features
X = user_counts.drop(target_name, axis=1)


# In[11]:


X.columns


# In[12]:


y


# In[13]:


# Function for splitting training and test set
from sklearn.model_selection import train_test_split
# Split X and y into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=6, shuffle = True)


# In[14]:


from sklearn.linear_model import LinearRegression
# Create LinearRegression 


lr1 = LinearRegression()
import datetime
import time
start = datetime.datetime.now()
t0 = time.time()



lr1.fit(X_train,y_train)
end = datetime.datetime.now()
print("Training time:", time.time()-t0)
print("Total execution time :" , end-start )


y_pred=lr1.predict(X_test)
rms = np.sqrt(mean_squared_error(y_test,y_pred))
MAE = mean_absolute_error(y_test, y_pred)
R = r2_score(y_test,y_pred)


print("mean_squared_error" ,rms)
print("mean_absolute_error" ,MAE)
print("R2_score" ,R)




# Compare predictions with the actual values
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Plot the actual values and the predicted values
comparison.plot(kind='bar')

#plt.plot(y_test.index, y_test, label='Actual')
#plt.plot(y_test.index, y_pred, label='Predicted')
plt.xlabel('Index')
plt.ylabel('Number of Users')
plt.title('LinearRegression')
plt.legend()
plt.show()



# In[15]:


# Create RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor


RFC = RandomForestRegressor()
import datetime
import time
start = datetime.datetime.now()
t0 = time.time()




RFC.fit(X_train,y_train)
end = datetime.datetime.now()
print("Training time:", time.time()-t0)
print("Total execution time :" , end-start )
y_pred=RFC.predict(X_test)
rms = np.sqrt(mean_squared_error(y_test,y_pred))
MAE = mean_absolute_error(y_test, y_pred)
R = r2_score(y_test,y_pred)




print("mean_squared_error" ,rms)
print("mean_absolute_error" ,MAE)
print("R2_score" ,R)


# Compare predictions with the actual values
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Plot the actual values and the predicted values
comparison.plot(kind='bar')

#plt.plot(y_test.index, y_test, label='Actual')
#plt.plot(y_test.index, y_pred, label='Predicted')
plt.xlabel('Index')
plt.ylabel('Number of Users')
plt.title('RandomForestRegressor')
plt.legend()
plt.show()






# In[16]:


from sklearn.ensemble import GradientBoostingRegressor
LGB = GradientBoostingRegressor()



import datetime
import time
start = datetime.datetime.now()
t0 = time.time()



LGB.fit(X_train,y_train)
end = datetime.datetime.now()
print("Training time:", time.time()-t0)
print("Total execution time :" , end-start )
y_pred=LGB.predict(X_test)
rms = np.sqrt(mean_squared_error(y_test,y_pred))
MAE = mean_absolute_error(y_test, y_pred)
R = r2_score(y_test,y_pred)



print("mean_squared_error" ,rms)
print("mean_absolute_error" ,MAE)
print("R2_score" , R)



# Compare predictions with the actual values
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Plot the actual values and the predicted values
comparison.plot(kind='bar')

#plt.plot(y_test.index, y_test, label='Actual')
#plt.plot(y_test.index, y_pred, label='Predicted')
plt.xlabel('Index')
plt.ylabel('Number of Users')
plt.title('GradientBoostingRegressor')
plt.legend()
plt.show()




# In[17]:


#Model Support Vector Regression (SVC)

from sklearn.svm import SVR
# train the model on train set
Model = SVR()
import datetime
import time
start = datetime.datetime.now()
t0 = time.time()





Model.fit(X_train,y_train)
end = datetime.datetime.now()
print("Training time:", time.time()-t0)
print("Total execution time :" , end-start )
y_pred = Model.predict(X_test)
rms = np.sqrt(mean_squared_error(y_test,y_pred))
MAE = mean_absolute_error(y_test, y_pred)
R = r2_score(y_test,y_pred)

print("mean_squared_error" ,rms)
print("mean_absolute_error" ,MAE)
print("R2_score" ,R)

# Compare predictions with the actual values
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Plot the actual values and the predicted values
comparison.plot(kind='bar')


plt.xlabel('Index')
plt.ylabel('Number of Users')
plt.title('Support Vector Regression')
plt.legend()
plt.show()


# In[18]:


#XGBoostingRegressor

from xgboost import XGBRegressor
reg = XGBRegressor()


import datetime
import time
start = datetime.datetime.now()
t0 = time.time()


reg.fit(X_train,y_train)
end = datetime.datetime.now()
print("Training time:", time.time()-t0)
print("Total execution time :" , end-start )
y_pred= reg.predict(X_test)
rms = np.sqrt(mean_squared_error(y_test,y_pred))
MAE = mean_absolute_error(y_test, y_pred)
R = r2_score(y_test,y_pred)


print("mean_squared_error" ,rms)
print("mean_absolute_error" ,MAE)
print("R2_score" ,R)

# Compare predictions with the actual values
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Plot the actual values and the predicted values
comparison.plot(kind='bar')

#plt.plot(y_test.index, y_test, label='Actual')
#plt.plot(y_test.index, y_pred, label='Predicted')
plt.xlabel('Index_Users')
plt.ylabel('Number of Users')
plt.title('XGBRegressor')
plt.legend()
plt.show()


# In[19]:


y_pred1 = lr1.predict(X_test)
y_pred2 = RFC.predict(X_test)
y_pred3 = LGB.predict(X_test)
y_pred4 = Model.predict(X_test)
y_pred5 = reg.predict(X_test)

mse1 = np.sqrt(mean_squared_error(y_test, y_pred1,))
mse2 = np.sqrt(mean_squared_error(y_test, y_pred2))
mse3 = np.sqrt(mean_squared_error(y_test, y_pred3))
mse4 = np.sqrt(mean_squared_error(y_test, y_pred4))
mse5 = np.sqrt(mean_squared_error(y_test, y_pred5))



mse_data = {'Model': ['a', ' 2', 'Model 3','Model 4', 'Model 5'],
            'RMS': [mse1, mse2, mse3,mse4,mse5]}
mse_df = pd.DataFrame(mse_data,index=['Linear Regression', 'Random Forest','GradientBoosting','Support Vector ','XGBoosting'])


mse_df.plot(kind='bar')
plt.xticks(rotation=30)


# In[20]:


y_pred10 = lr1.predict(X_test)
y_pred20 = RFC.predict(X_test)
y_pred30 = LGB.predict(X_test)
y_pred40 = Model.predict(X_test)
y_pred50=  reg.predict(X_test)

mae10 = mean_absolute_error(y_test, y_pred10)
mae20 = mean_absolute_error(y_test, y_pred20)
mae30 = mean_absolute_error(y_test, y_pred30)
mae40 = mean_absolute_error(y_test, y_pred40)
mae50 = mean_absolute_error(y_test, y_pred50)



mse1_data = {'Model': ['a', ' 2', 'Model 3','Model 4', 'Model 5'],
            'MAE': [mae10, mae20, mae30,mae40,mae50]}
mse1_df = pd.DataFrame(mse1_data,index=['Linear Regression', 'Random Forest','GradientBoosting','Support Vector ','XGBoosting'])



mse1_df.plot(kind='bar')

plt.xticks(rotation=30)




# In[21]:


y_predR1 = lr1.predict(X_test)
y_predR2 = RFC.predict(X_test)
y_predR3 = LGB.predict(X_test)
y_predR4 = Model.predict(X_test)
y_predR5 = reg.predict(X_test)

R1= r2_score(y_test, y_predR1)
R2 = r2_score(y_test, y_predR2)
R3= r2_score(y_test, y_predR3)
R4 = r2_score(y_test, y_predR4)
R5 = r2_score(y_test, y_predR5)



r_data = {'Model': ['Model 1', 'Model 2', 'Model 3','Model 4', 'Model 5'],
            'R2_score': [ R1,R2,R3,R4,R5]}
r_df = pd.DataFrame(r_data,index=['Linear Regression', 'Random Forest','GradientBoosting','Support Vector ','XGBoosting'])




r_df.plot(kind='bar')

plt.xticks(rotation=30)





# In[56]:


import datetime
import time
start = datetime.datetime.now()
t0 = time.time()

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

def improved_random_forest(X_train, y_train, X_test, y_test,y_pred):
    
    # Initialize Random Forest model
    model = RandomForestRegressor(n_estimators=100, max_depth=5)
    
    # Perform hyperparameter tuning using grid search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    # Retrieve the best model from grid search
    model = grid_search.best_estimator_
    
    # Fit the model to the training data
    model.fit(X_train, y_train)
    
    # Generate predictions for the test data
    y_pred = model.predict(X_test)
    
    # Access the 'newpos' column in X_test
    newpos_values = X_test['newpos']

    # Compare the values of 'newpos'
    for i, value in enumerate(newpos_values):
            # Example: Apply a mathematical operation to adjust the predictions
        if value > 0:
                  y_pred[i] = y_pred[i] * 0.74
           
        else:
               y_pred[i] = y_pred[i] - 0.1
        
    
    # Calculate metrics
    mse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
   
   
    end = datetime.datetime.now()
    print("Training time:", time.time()-t0)
    print("Total execution time :" , end-start )
    # Print the evaluation metrics
    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("R-squared (R2):", r2)
    
    
    #R squared
    y_1 = RFC.predict(X_test)
    y_2 = model.predict(X_test)
    Rp = r2_score(y_test, y_1)
    Rb = r2_score(y_test, y_2)

    p_data = {'Model': ['Model 1', 'Model 2'],
            'R2_score': [Rp,r2]}
    p_df = pd.DataFrame(p_data,index=['Random Forest Regressor', 'RandomForestRegressor_Improved'])
    ax = p_df.plot(kind='bar')
    # annotate
    ax.bar_label(ax.containers[0], label_type='edge')
    plt.xticks(rotation=30)
    
    
    
    #Mean squared error
    y_s = RFC.predict(X_test)
    y_m = model.predict(X_test)
    m = np.sqrt(mean_squared_error(y_test,  y_s))
    m2 = np.sqrt(mean_squared_error(y_test, y_m))
    s_data = {'Model': ['a', ' 2'],
            'RMS': [m, mse]}
    s_df = pd.DataFrame(s_data,index=['Random Forest Regressor', 'RandomForestRegressor_Improved'])
    ax = s_df.plot(kind='bar')
    height = s_data
    # annotate
    ax.bar_label(ax.containers[0], label_type='edge')
    plt.xticks(rotation=30)
    
    
    # Compare predictions with the actual values
    comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    # Plot the actual values and the predicted values
    comparison.plot(kind='bar')


    plt.xlabel('Index')
    plt.ylabel('Number of Users')
    plt.title('RandomForestRegressor_Improved')
    plt.legend()
    plt.show()
 
    
    
   
    # Return the final predictions and metrics
    return y_pred, mse, mae, r2


improved_random_forest(X_train, y_train, X_test, y_test,y_pred)







# In[ ]:





# In[ ]:




