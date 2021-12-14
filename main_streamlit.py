#Imports
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns

st.markdown(
      '''
      <style>
      .reportview-container .main .block-container{{
        max-width: 90%;
        padding-top: 5rem;
        padding-right: 5rem;
        padding-left: 5rem;
        padding-bottom: 5rem;
    }}
    img{{
    	max-width:40%;
    	margin-bottom:40px;
    }}
      </style>
      ''',
      unsafe_allow_html=True
)
###################################################

header_cont = st.container()
stat_cont = st.container()


with header_cont: # Creating a container that keep in the header and introduction
      st.title('Creditcard Fraud Detector WebApp')
    

      
      col_1, col_2 = st.columns([1,2])
      with col_1:
            st.subheader('Introduction')

      with col_2:
            st.write('''In this data science project I would be building a credit card 
                        fraud detection model and this web app is used to show the walk through of the project
                         from the balancing of the dataset to the model buiding. out of the five models used while
                        buliding the model three were used in this webapp Namely: [Decision Tree Classfier],
                        [Random Forest Classifier] and [Extreme Gradient Boosting].

      ''')

@st.cache
def get_data(filename, file_drop, axis):  # this function get the data 
            dt = pd.read_csv(filename)    # Instead of reading datas multiple times
            if file_drop and axis == None:
                  pass 
            else:
                  dt.drop(file_drop, axis=axis, inplace=True)
            return dt

with stat_cont:
      st.header("Datasets")

# The first dataset
      st.subheader('First Dataset')
      data = (get_data('https://github.com/kinematic2002/Streamlit_CFDM_app/blob/0cc346ce590ac81f021367afd7c09e75ba9cfc6b/data/sampled_creditcard.csv', "Unnamed: 0", 1)) # first dataset
      # Counting the values of the Target variables
      data_vc = pd.DataFrame(data['Class'].value_counts())
      data_list = data.columns.to_list() # listing out the variables of the dataset
      st.write(f'''The first Data set provided has the shape {data.shape} and contained the variables
            {data_list} but was not used to build the model because of its imbalanced nature were the class
            variable(Target) had 2000 valid transctions and 492 fraudlent transactions which will not be 
            enough to predict and get the best accuracy for this model.
      ''')
      # showing the dataset and visualizing the values of the target variable
      st.write(pd.DataFrame(data).head(50))
      st.bar_chart(data_vc, width=600, height=400)

      # getting the number of the fraudlent transaction and Valid transaction.....
      fraud = data[data['Class']==1]
      normal = data[data['Class'] == 0]
      st.write ('Fraudlent Transactions of the first dataset(1): {}'.format(len(fraud)))
      st.write('Non-Fraudlent Transactions of the first dataset(0): {}'.format(len(normal)))
      st.write(f'Number Classes: {len(np.unique(data["Class"]))}')

# Second Dataset
      st.subheader('Second Dataset')

      data_1 = get_data('https://github.com/kinematic2002/Streamlit_CFDM_app/blob/5c82c0c8d8b7f13a1ad3f84c4be8fc6d840a262e/data/BalancedCreditCardDataset.csv', "Unnamed: 0", 1) # second dataset
      data_1_vc = pd.DataFrame(data_1['Class'].value_counts())
      data_1_list = data_1.columns.to_list()
      # write up for the second dataset
      st.write(f'''The second Data set provided has the shape {data_1.shape} and contained the variables
            {data_1_list} which was used to build the model because of it was balanced from the first dataset
            the class variable(Target) had 2000 valid transctions and 2000 fraudlent transactions 
            which will be enough to predict and get the best accuracy for this model.
      ''')

      # showing the dataset and visualizing the values of the target variable
      st.write(pd.DataFrame(data_1).head(50))
      st.bar_chart(data_1_vc, width=600, height=400)


      # getting the number of the fraudlent transaction and Valid transaction.....
      fraud = data_1[data_1['Class']==1]
      normal = data_1[data_1['Class'] == 0]
      st.write ('Fraudlent Transactions of the first dataset(1): {}'.format(len(fraud)))
      st.write('Non-Fraudlent Transactions of the first dataset(0): {}'.format(len(normal)))
      st.write(f'Number Classes: {len(np.unique(data_1["Class"]))}')

# Side bar for predicting
page = st.sidebar.selectbox('Select Your Model', ['Decision Tree', 'Random Forest Classifier', 'Extreme Gradient Boosting '])
st.cache(page)
st.sidebar.write( '''
NOTE: Values used to predict are them same as the values in the datasets.
''')
st.sidebar.write("Now, Let's Get Predicting!!!!!")
# variables in the datasets
# Time Variable
time = st.sidebar.slider('Time', min_value=1000.000000, max_value=200000.0000, value=1000.0000, step=1000.0500)
# V1 - V27 variables
v1 = st.sidebar.slider('V1', min_value=-100.0000, max_value=100.0000, value=-80.0000, step=0.0500)
v2 = st.sidebar.slider('V2', min_value=-100.0000, max_value=100.0000, value=-80.0000, step=0.0500)
v3 = st.sidebar.slider('V3', min_value=-100.0000, max_value=100.0000, value=-80.0000, step=0.0500)
v4 = st.sidebar.slider('V4', min_value=-100.0000, max_value=100.0000, value=-80.0000, step=0.0500)
v5 = st.sidebar.slider('V5', min_value=-100.0000, max_value=100.0000, value=-80.0000, step=0.0500)
v6 = st.sidebar.slider('V6', min_value=-100.0000, max_value=100.0000, value=-80.0000, step=0.0500)
v7 = st.sidebar.slider('V7', min_value=-100.0000, max_value=100.0000, value=-80.0000, step=0.0500)
v8 = st.sidebar.slider('V8', min_value=-100.0000, max_value=100.0000, value=-80.0000, step=0.0500)
v9 = st.sidebar.slider('V9', min_value=-100.0000, max_value=100.0000, value=-80.0000, step=0.0500)
v10 = st.sidebar.slider('V10', min_value=-100.0000, max_value=100.0000, value=-80.0000, step=0.0500)
v11 = st.sidebar.slider('V11', min_value=-100.0000, max_value=100.0000, value=-80.0000, step=0.0500)
v12 = st.sidebar.slider('V12', min_value=-100.0000, max_value=100.0000, value=-80.0000, step=0.0500)
v13 = st.sidebar.slider('V13', min_value=-100.0000, max_value=100.0000, value=-80.0000, step=0.0500)
v14 = st.sidebar.slider('V14', min_value=-100.0000, max_value=100.0000, value=-80.0000, step=0.0500)
v15 = st.sidebar.slider('V15', min_value=-100.0000, max_value=100.0000, value=-80.0000, step=0.0500)
v16 = st.sidebar.slider('V16', min_value=-100.0000, max_value=100.0000, value=-80.0000, step=0.0500)
v17 = st.sidebar.slider('V17', min_value=-100.0000, max_value=100.0000, value=-80.0000, step=0.0500)
v18 = st.sidebar.slider('V18', min_value=-100.0000, max_value=100.0000, value=-80.0000, step=0.0500)
v19 = st.sidebar.slider('V19', min_value=-100.0000, max_value=100.0000, value=-80.0000, step=0.0500)
v20 = st.sidebar.slider('V20', min_value=-100.0000, max_value=100.0000, value=-80.0000, step=0.0500)
v21 = st.sidebar.slider('V21', min_value=-100.0000, max_value=100.0000, value=-80.0000, step=0.0500)
v22 = st.sidebar.slider('V22', min_value=-100.0000, max_value=100.0000, value=-80.0000, step=0.0500)
v23 = st.sidebar.slider('V23', min_value=-100.0000, max_value=100.0000, value=-80.0000, step=0.0500)
v24 = st.sidebar.slider('V24', min_value=-100.0000, max_value=100.0000, value=-80.0000, step=0.0500)
v25 = st.sidebar.slider('V25', min_value=-100.0000, max_value=100.0000, value=-80.0000, step=0.0500)
v26 = st.sidebar.slider('V26', min_value=-100.0000, max_value=100.0000, value=-80.0000, step=0.0500)
v27 = st.sidebar.slider('V27', min_value=-100.0000, max_value=100.0000, value=-80.0000, step=0.0500)
v28 = st.sidebar.slider('V28', min_value=-100.0000, max_value=100.0000, value=-80.0000, step=0.0500)
# Amount variable
amount = st.sidebar.slider('Amount', min_value=0.0000, max_value=1000000.0000, value=100.0000, step=200.0000)
# Predict Button
predict_button = st.sidebar.button('Predict')
names = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16',
         'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
X = np.array([time,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,amount])
data = pd.DataFrame([X], columns=names).astype("int")

# Model Building
# imports for model algorithms and model evaluation
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
# Mean Squared error
from sklearn.metrics import mean_squared_error



# separating the feature from the target variable.......by slicing the dataset
x = data_1.iloc[:, :-1]
y = data_1.iloc[:, -1]

if page == 'Decision Tree':
      # Decision Tree Classifier
      dtree_2 = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5)
      dtree_2.fit(x,y.values)
      if predict_button:
            ypred_dtree_2 = dtree_2.predict(data)
            
            if st.sidebar.write(ypred_dtree_2[0]) == 1:
                  st.sidebar.subheader('Beware Fraudlent Transaction!!!!')
            else:
                  st.sidebar.subheader('Valid Transaction')
      else:
            st.sidebar.write('Click to predict')

elif page == 'Random Forest Classifier':
      # Random Forest Classifier
      rf = RandomForestClassifier(n_estimators= 150, min_samples_split=5, min_samples_leaf=5)
      rf.fit(x, y.values)
      if predict_button:
            ypred_rf = rf.predict(data)
            if st.sidebar.write(ypred_rf[0]) == 1:
                  st.sidebar.subheader('Beware Fraudlent Transaction!!!!')
            else:
                  st.sidebar.subheader('Valid Transaction')
      else:
            st.sidebar.write('Click to predict')

else:
# XGBoost Classfier
      xgbc = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=True, n_estimators=100, n_jobs=5)
      xgbc.fit(x, y.values)
      if predict_button:
            ypred_xgb = xgbc.predict(data)
            if st.sidebar.write(ypred_xgb[0]) == 1:
                  st.sidebar.subheader('Beware Fraudlent Transaction!!!!')
            else:
                  st.sidebar.subheader('Valid Transaction')
      else:
            st.sidebar.write('Click to predict')
     
