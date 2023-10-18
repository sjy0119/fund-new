import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
   df=pd.read_csv("基金经理")
   return df
df=load_data()

manager_name=df['基金经理名称']


option = st.selectbox(
    '请选择基金经理的姓名',
    tuple(manager_name))

id=df.loc[df['基金经理名称']==option]['基金经理id'].values[0]

def ld_data():
    data_info=pd.read_html(f'http://fund.eastmoney.com/manager/{id}.html')
    return data_info

data_info=ld_data()

fund_all=data_info[1]
fund_all['基金代码']=fund_all['基金代码'].apply(lambda x: '00000'+str(x)).apply(lambda x: x[-6:])
fund_pre=data_info[-1]
fund_pre['基金代码']=fund_pre['基金代码'].apply(lambda x: '00000'+str(x)).apply(lambda x: x[-6:])

if option:
    info=df.loc[df['基金经理名称']==option]['基金经理简介'].values[0]
    info1=info.split('。')

    mag=df.loc[df['基金经理名称']==option]['照片链接'].values[0]


    col1, col2= st.columns(2)

    with col1:
        st.header(option)
        st.image(mag)

    with col2:
        st.header("基金经理基本信息")
        st.write(info1)
    with st.container():

        st.header(f"{option}管理基金一览")

        
        st.dataframe(fund_all,hide_index=True)

    with st.container():

        st.header(f"{option}现任基金业绩与排名详情")

        
        st.dataframe(fund_pre,hide_index=True)


