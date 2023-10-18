import streamlit as st
import time
import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta,date
import plotly as py
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio
import seaborn as sns
import tushare as ts
pro = ts.pro_api('8e812052c92d7a829f0e3b0197d248e48bb2ba3efbbaa60f505e6852')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] =False #减号unicode编码
#st.set_page_config(page_icon="📈",)
st.markdown("# 基金净值信息数据查询")
st.sidebar.header("基金净值信息数据查询")
st.write(
    """在该模块之中，大家可以选择性输入开始和结束的时间，以及基金代码和所要查询的
    基金净值的类型，同时该模块的两个按钮一个展示基金净值的走势情况，一个展示基金净值的原始数据"""
)
start_date = st.date_input(
    "请选择开始日期",
    date(2020,2,9))
#st.write('开始日期:', start_date)
开始=str(start_date)[:4]+str(start_date)[5:7]+str(start_date)[8:]
end_date = st.date_input(
    "请选择结束日期",
    date(2021,5,9))
#st.write('结束日期:',end_date)
结束=str(end_date)[:4]+str(end_date)[5:7]+str(end_date)[8:]
code=st.text_input('请输入基金代码例如000001.OF')

hg = st.text_input(
        "请输入净值名称例如单位净值 👇",
)
if hg:
    leix=str(hg)
    if leix=='单位净值':
        df = pro.fund_nav(ts_code=code,start_date=开始,end_date=结束,fields=['ts_code','nav_date','unit_nav'])
        df['nav_date']=pd.to_datetime(df['nav_date'])
        df=df.sort_values(by='nav_date',ignore_index=True)
        df1=df.rename(columns={'unit_nav':'value'})
            #y=df[['day','net_value']].set_index('day').plot(figsize=(9,6),grid=True)#基金净值静态可视化
    elif leix=='累计净值':
        df = pro.fund_nav(ts_code=code,start_date=开始,end_date=结束,fields=['ts_code','nav_date','accum_nav'])
        df['nav_date']=pd.to_datetime(df['nav_date'])
        df=df.sort_values(by='nav_date',ignore_index=True)
        df1=df.rename(columns={'accum_nav':'value'})
            #y=df[['day','sum_value']].set_index('day').plot(figsize=(9,6),grid=True)#基金净值静态可视化
    else:
        df = pro.fund_nav(ts_code=code,start_date=开始,end_date=结束,fields=['ts_code','nav_date','adj_nav'])
        df['nav_date']=pd.to_datetime(df['nav_date'])
        df=df.sort_values(by='nav_date',ignore_index=True)
        df1=df.rename(columns={'adj_nav':'value'})
            #y=df[['day','net_value']].set_index('day').plot(figsize=(9,6),grid=True)#基金净值静态可视化

if st.checkbox('绘制净值走势图'):
    st.line_chart(df1,x='nav_date',y='value')

if st.checkbox('展示原始数据'):
    st.subheader('原始数据')
    st.dataframe(df1)
