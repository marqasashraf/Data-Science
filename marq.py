#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import zipfile
import concurrent.futures
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ## This code reads multiple zip files that contain multiple CSV files of sensor data, extracts the required data from them, and combines them into a single pandas DataFrame. The extracted data includes ACC, BVP, EDA, HR, IBI, and TEMP. The extract_data function takes in a list of zip file names, iterates through the files, and for each file, extracts the required data and renames the columns. The extracted data is then joined to create a single DataFrame. The function returns this DataFrame after dropping any rows that contain missing values. The main part of the code uses the glob module to get a list of all the folders that contain the zip files. It then iterates over these folders and extracts data from all the zip files in each folder using the extract_data function. The resulting DataFrames are concatenated into a single DataFrame, and any rows with temporary IBI values are dropped. The resulting DataFrame is returned.

# In[2]:


import pandas as pd
import glob
import os
import zipfile

def extract_data(files):
    joined_list = []
    for file in files:
        with zipfile.ZipFile(file) as zf:
            zf.extractall()
            sensor_coords = pd.read_csv('ACC.csv', index_col=None, header=0)
            BVP = pd.read_csv('BVP.csv', index_col=None, header=0)
            eda = pd.read_csv('EDA.csv', index_col=None, header=0)
            heart_rates = pd.read_csv('HR.csv', index_col=None, header=0)
            ibi = pd.read_csv('IBI.csv', index_col=None, header=0) if os.path.getsize('IBI.csv') > 0 else None
            temp = pd.read_csv('TEMP.csv', index_col=None, header=0)

            sensor_coords = sensor_coords.rename(columns={sensor_coords.columns[0]: 'X_AXIS', sensor_coords.columns[1]: 'Y_AXIS', sensor_coords.columns[2]: 'Z_AXIS'})
            BVP = BVP.rename(columns={BVP.columns[0]: 'BVP'})
            eda = eda.rename(columns={eda.columns[0]: 'EDA'})
            heart_rates = heart_rates.rename(columns={heart_rates.columns[0]: 'HR'})
            if ibi is not None:
                ibi = ibi.rename(columns={ibi.columns[0]: 'IBI', ibi.columns[1]: 'IBI_TWO'})
            
            temp = temp.rename(columns={temp.columns[0]: 'TEMP'})

            if ibi is not None:
                joined = sensor_coords.join(BVP).join(eda).join(heart_rates).join(ibi).join(temp).dropna()
            else:
                joined = sensor_coords.join(BVP).join(eda).join(heart_rates).dropna()
                joined['IBI'] = 'Temporary'
                joined['IBI_TWO'] = 'Temporary'
                joined = joined.join(temp)

            joined_list.append(joined)

    df = pd.concat(joined_list).reset_index(drop=True)
    df = df[~(df['IBI'] == 'Temporary')]
    return df

folders = glob.glob(os.path.join("Data/*"))
df = pd.concat([extract_data(glob.glob(os.path.join(folder, '*.zip'))) for folder in folders]).reset_index(drop=True)


# In[5]:


# Removing Unnamed column from the dataset
df = pd.read_csv('processed.csv')
df = df.drop('Unnamed: 0', axis=1)
df


# In[6]:


df.info()


# ## This code is performing two operations on a pandas DataFrame called df. The first operation is filtering out rows where the value in the INTERVAL column is equal to the string ' IBI'. The tilde (~) operator is used to negate the Boolean condition in the parentheses, so that the resulting DataFrame includes only the rows where the condition is False. The second operation is converting the INTERVAL column from an object (presumably a string) to a float data type. The .astype() method is used to change the data type of the column to float. Overall, this code is cleaning the INTERVAL column of the df DataFrame by removing rows with a certain value, and then converting the remaining values to float data type.

# In[9]:


# Converting object to float
df = df[~(df['INTERVAL'] == ' IBI')]
df['INTERVAL'] = df['INTERVAL'].astype(float)


# In[12]:


df = df[~(df['INTERVAL'] == ' INTERVAL')]
df['INTERVAL'] = df['INTERVAL'].astype(float)


# In[13]:


df.dtypes


# In[14]:


# Statistics of the data
df.describe()


# ## This code calculates the 30th percentile (quantile) of three variables 'X', 'Y', and 'Z' in the DataFrame 'df' using the NumPy library's 'quantile' function. It then uses these quantiles to create a new column in 'df' called 'target' using the 'np.where' function. The 'target' column contains binary values, either 1 or 0, depending on whether the corresponding row values in 'X', 'Y', 'Z', 'BVP', and 'HR' are greater than the calculated quantiles. Specifically, if the value of 'X' is greater than the 'acc_quantile', the value of 'Y' is greater than the 'acc_quantile', the value of 'Z' is greater than the 'acc_quantile', the value of 'BVP' is greater than the 'bvp_quantile', and the value of 'HR' is greater than the 'hr_quantile', the corresponding value in the 'target' column will be set to 1; otherwise, it will be set to 0.

# In[16]:


# This is use for labeling. According to paper i identify these columns and set the threshold to 30 percent. if values exceeds 30 percent then it will be 1 otherwise 0
import numpy as np

acc_quantile = np.quantile(df['X'], 0.30)
bvp_quantile = np.quantile(df['Y'], 0.30)
hr_quantile = np.quantile(df['Z'], 0.30)

df['target'] = np.where((df['X'] > acc_quantile) &
                          (df['Y'] > acc_quantile) &
                          (df['Z'] > acc_quantile) &
                          (df['BVP'] > bvp_quantile) &
                          (df['HR'] > hr_quantile), 1, 0)


# ## This code is performing downsampling on a Pandas DataFrame df which has a column named 'target'. First, it is creating two new DataFrames: df_Z which contains all rows from df where the value in the 'target' column is 0, and df_O which contains all rows where the value in the 'target' column is 1. Then, it is downsampling the df_Z DataFrame to a size of 327000 rows. It does this by randomly selecting 327000 rows from df_Z using the .sample() method. Finally, it is concatenating the downsampled df_Z DataFrame with the df_O DataFrame using the pd.concat() method with ignore_index=True. This creates a new DataFrame with the same columns as df, where the first 327000 rows are from df_Z and the remaining rows are from df_O. The resulting DataFrame is returned and may be used for further analysis or modeling. The purpose of this downsampling is likely to address class imbalance in the target variable, where there are many more rows with a target value of 0 than 1. By downsampling the majority class, the resulting DataFrame has a more balanced distribution of target values, which can improve model performance for certain types of models.

# In[31]:


# Downsampling
df_Z = df[df['target'] == 0]
df_O = df[df['target'] == 1]

df_Z = df_Z.sample(327000)

df = pd.concat([df_Z,df_O],ignore_index = True)


# In[32]:


sns.countplot(x='target', data=df)
plt.title("Count of Labels")
plt.xlabel("0's and 1's")
plt.ylabel("Count")
plt.show()


# In[33]:


df_Z.shape, df_O.shape


# In[34]:


df['target'].value_counts()


# In[28]:


df.corr()


# In[29]:


df.corr().transpose()


# In[30]:


sns.heatmap(df.corr().transpose(), cmap='plasma')
plt.show()


# In[23]:


df.info()


# ## This code is performing several steps to create and train an LSTM (Long Short-Term Memory) model for binary classification. First, it imports the StandardScaler class from the sklearn.preprocessing module and uses it to normalize the input data in a pandas DataFrame called "df". This step scales the data so that each feature has a mean of 0 and a standard deviation of 1, making it easier for the model to learn from. Then, it splits the data into training and testing sets using the train_test_split function from the sklearn.model_selection module. The X variable is assigned the features from the DataFrame, except for the last column, which contains the target variable. The y variable is assigned the target variable. Next, the data is converted to arrays using the numpy module. The training and testing arrays are then reshaped into the input format required for an LSTM model, which expects a 3D input shape of (samples, timesteps, features). The LSTM layer is added to the Sequential model, along with a Dense output layer with a sigmoid activation function. Finally, the model is compiled with binary cross-entropy loss, the Adam optimizer, and accuracy as the evaluation metric. The model is trained using the fit method, with the training data, validation data, and other hyperparameters such as the number of epochs and batch size specified. After training, the model is ready to make predictions on new data.

# In[35]:


from sklearn.preprocessing import StandardScaler

# Normalize the data in df
scaler = StandardScaler()
df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X = df[df.columns[:-1]]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert the data to arrays
import numpy as np

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Reshape the data for LSTM input
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Define and train the LSTM model
from keras.models import Sequential
from keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))


# In[19]:


pd.DataFrame(model.history.history).plot()


# In[21]:


from sklearn.metrics import accuracy_score
# evaluate the model on the testing data
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy score: {accuracy}")


# In[ ]:




