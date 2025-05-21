import streamlit as st
import pandas as pd
import numpy as np
import pickle


# Judul
st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE; font-family: Arial, sans-serif;'>
        📉 Customer Churn Prediction 
    </h1>
    """,
    unsafe_allow_html=True
)
st.write("")
st.write("")
st.write("")


# Sidebar
st.sidebar.header("Please Input Customer Feature")
def inputData():
       Tenure = st.sidebar.number_input("Tenure", min_value=0, value=0)
       WarehouseToHome=st.sidebar.number_input('Home To Warehouse Distance', min_value=0, max_value=120, value=0)
       NumberOfDeviceRegistered=st.sidebar.selectbox('Device Registered', [1, 2, 3, 4, 5, 6])
       DaySinceLastOrder=st.sidebar.number_input('Days Since Last Order', min_value=0, max_value=45, value=0)
       CashbackAmount=st.sidebar.number_input('Cash Back Amount', min_value=0, value=0, max_value=325)
       PreferedOrderCat=st.sidebar.selectbox('Prefer Order Category', ['Laptop & Accessory', 'Mobile', 'Fashion', 'Others',
              'Mobile Phone', 'Grocery'])
       SatisfactionScore=st.sidebar.selectbox("Statisfaction Score", [1, 2, 3, 4, 5])
       MaritalStatus=st.sidebar.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
       NumberOfAddress=st.sidebar.selectbox('Number of Registered Addresses', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
       Complain=st.sidebar.radio('Have Complain or Not', ['Yes', 'No'])

       df = pd.DataFrame()
       df["Tenure"] = [Tenure]
       df["WarehouseToHome"] = [WarehouseToHome]
       df["NumberOfDeviceRegistered"] = [NumberOfDeviceRegistered]
       df["PreferedOrderCat"] = [PreferedOrderCat]
       df["SatisfactionScore"] = [SatisfactionScore]
       df["MaritalStatus"] = [MaritalStatus]
       df["NumberOfAddress"]=[NumberOfAddress]
       df["Complain"] = [1 if Complain=='Yes' else 0]
       df["DaySinceLastOrder"] = [DaySinceLastOrder]
       df["CashbackAmount"]=[CashbackAmount]
       return df

# membuat dataframe berdasarkan input
df_predict = inputData()
df_predict.index = [0]

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_button = st.button("🔍 Predict", use_container_width=True)
st.write('')
st.write('')
st.write('')


#Model Load
if predict_button:
       col1, col2, col3=st.columns(3)
       model_loaded = pickle.load(open("D:\Data Science\Latihan-Modul3\Capstone\model_customer_churn.sav", "rb"))
       predict = model_loaded.predict(df_predict)
       proba=model_loaded.predict_proba(df_predict)
       if predict == 0:
              col1.header("Not Churn")
              col3.metric(label='', value=f"{proba[0][0]:.2f}%")
              st.success(f"✅ Customer will not churn (Probability: {proba[0][0]:.2f}%)")
              
       else:
              col1.header("CHURN")
              col3.metric(label='', value=f"{proba[0][1]:.2f}%")
              st.success(f"⚠️ Customer will churn (Probability: {proba[0][1]:.2f}%)")
st.write('')
st.write('')
st.write('')

st.sidebar.write('Upload Your File')

uploaded_file = st.sidebar.file_uploader("CSV File", type="csv")

file=True

if uploaded_file is not None:
    with st.expander('Dataset'):
       dfUp= pd.read_csv(uploaded_file, index_col=False)
       st.write("📄 Data Preview:")
       st.write(dfUp)
       file=False

predictDataset=st.sidebar.button("🔍 Predict Dataset")

if predictDataset :
       if file==False:
              model_loaded = pickle.load(open("D:\Data Science\Latihan-Modul3\Capstone\model_customer_churn.sav", "rb"))
              predDataset = model_loaded.predict(dfUp)
              probaDataset=model_loaded.predict_proba(dfUp)
              dfUp['Churn']=predDataset
              dfUp['Churn Proba (%)']=np.round(probaDataset[:, 1]*100, 2)
              with st.expander("Predicted Dataset"):
                     st.write(dfUp)
              with st.expander("Customer Churn Data"):
                     dfUpChurn=dfUp[dfUp['Churn']==1]
                     st.write(dfUpChurn.sort_values('Churn Proba (%)', ascending=False))
                     st.success(f"⚠️ {len(dfUpChurn)} Customer will churn")
       else :
              st.sidebar.write('⚠️ No File Uploaded !')





