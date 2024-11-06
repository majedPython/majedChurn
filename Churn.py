import pandas as pd
import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import pickle
import warnings

warnings.filterwarnings('ignore')

st.title("Churn Prediction")

stat={'TENURE': '', 'MONTANT':'', 'FREQUENCE_RECH':'', 'REVENUE':'',
       'ARPU_SEGMENT':'', 'FREQUENCE':'', 'DATA_VOLUME':'', 'ON_NET':'', 'ORANGE':'', 'TIGO':'',
       'REGULARITY':'', 'TOP_PACK':'', 'FREQ_TOP_PACK':'', 'DAKAR':0.0, 'DIOURBEL':0.0,
       'FATICK':0.0, 'KAFFRINE':0.0, 'KAOLACK':0.0, 'KEDOUGOU':0.0,
       'KOLDA':0.0, 'LOUGA':0.0, 'MATAM':0.0, 'SAINT-LOUIS':0.0,
       'SEDHIOU':0.0, 'TAMBACOUNDA':0.0, 'THIES':0.0,
       'ZIGUINCHOR':0.0}

with open('le.pkl', 'rb') as f:
    top_pack = pickle.load(f)

with open('le2.pkl', 'rb') as f:
    tenure = pickle.load(f)


st.text("Input subscriber statistics")
Ten= st.selectbox(
    'TENURE',
    ('D 3-6 month', 'E 6-9 month', 'F 9-12 month', 'G 12-15 month','H 15-18 month', 'I 18-21 month', 'J 21-24 month', 'K > 24 month'))
print(Ten)
if Ten:
    stat['TENURE']=int(tenure.transform([Ten])[0])


TOP_PACK= st.selectbox(
    'TOP_PACK',
    ('200=Unlimited1Day', 'All-net 1000=5000;5d',
       'All-net 1000F=(3000F On+3000F Off);5d', 'All-net 300=600;2d',
       'All-net 5000= 20000off+20000on;30d',
       'All-net 500F =2000F_AllNet_Unlimited',
       'All-net 500F=1250F_AllNet_1250_Onnet;48h',
       'All-net 500F=2000F;5d', 'All-net 600F= 3000F ;5d',
       'CVM_on-net bundle 500=5000', 'Data: 100 F=40MB,24H',
       'Data: 200 F=100MB,24H', 'Data:1000F=2GB,30d', 'Data:1000F=5GB,7d',
       'Data:1500F=3GB,30D', 'Data:1500F=SPPackage1,30d',
       'Data:150F=SPPackage1,24H', 'Data:200F=Unlimited,24H',
       'Data:3000F=10GB,30d', 'Data:300F=100MB,2d', 'Data:30Go_V 30_Days',
       'Data:490F=1GB,7d', 'Data:500F=2GB,24H', 'Data:50F=30MB_24H',
       'Data:700F=1.5GB,7d', 'Data:DailyCycle_Pilot_1.5GB',
       'Facebook_MIX_2D', 'IVR Echat_Daily_50F', 'Jokko_Daily',
       'Jokko_Monthly', 'Jokko_promo',
       'MIXT: 200mnoff net _unl on net _5Go;30d',
       'MIXT: 390F=04HOn-net_400SMS_400 Mo;4h\t',
       'MIXT: 590F=02H_On-net_200SMS_200 Mo;24h\t\t',
       'MIXT:500F= 2500F on net _2500F off net;2d', 'MROMO_TIMWES_OneDAY',
       'MROMO_TIMWES_RENEW', 'Mixt 250F=Unlimited_call24H',
       'On net 200F= 3000F_10Mo ;24H', 'On net 200F=Unlimited _call24H',
       'On-net 1000F=10MilF;10d', 'On-net 200F=60mn;1d',
       'On-net 500=4000,10d', 'On-net 500F_FNF;3d', 'Pilot_Youth1_290',
       'Pilot_Youth4_490', 'SUPERMAGIK_1000', 'SUPERMAGIK_5000',
       'Twter_U2opia_Daily', 'Twter_U2opia_Weekly',
       'VAS(IVR_Radio_Daily)', 'YMGX 100=1 hour FNF, 24H/1 month',
       'Yewouleen_PKG'))

if TOP_PACK:
    stat['TOP_PACK']=int(top_pack.transform([TOP_PACK])[0])

region = st.radio(
        'Choose the REGION',
        ('DAKAR', 'DIOURBEL',
       'FATICK', 'KAFFRINE', 'KAOLACK', 'KEDOUGOU',
       'KOLDA', 'LOUGA', 'MATAM', 'SAINT-LOUIS',
       'SEDHIOU', 'TAMBACOUNDA', 'THIES',
       'ZIGUINCHOR'))
if region:
    stat[region]=1.0


regularity=st.number_input(f"REGULARITY (integer)")
if regularity:
    stat['REGULARITY']=int(regularity)

list=['MONTANT', 'FREQUENCE_RECH', 'REVENUE',
       'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO', 'FREQ_TOP_PACK']
for f in list:
    X=st.number_input(f"{f} (float)")
    if X:
        stat[f]=float(X)
        
Region=['DAKAR', 'DIOURBEL','FATICK', 'KAFFRINE', 'KAOLACK', 'KEDOUGOU','KOLDA', 'LOUGA', 'MATAM', 'SAINT-LOUIS','SEDHIOU', 'TAMBACOUNDA', 'THIES','ZIGUINCHOR']

for i in Region:
    new='REGION_'+i
    stat[new]=stat.pop(i)

st.text('Summary of Subscriber data:')
df=pd.DataFrame(stat,index=[0])
st.table(df)

model = joblib.load('best_random_forest_model1.joblib')


if st.button("Prediction Churn/not Churn:"):
    churn=model.predict(df)
    pred=''
    if churn[0]==0: 
        pred='Not Churn'
    else:
        pred='Churn'
    st.text(f'This subscriber is probably will: {pred} ')


