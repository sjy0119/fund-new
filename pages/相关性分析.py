import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta,date
import plotly as py
import matplotlib.pyplot as plt
import seaborn as sns
import tushare as ts
import requests
import plotly.figure_factory as ff
pro = ts.pro_api('8e812052c92d7a829f0e3b0197d248e48bb2ba3efbbaa60f505e6852')
from py_mini_racer import py_mini_racer
from akshare.index.cons import (
    zh_sina_index_stock_payload,
    zh_sina_index_stock_url,
    zh_sina_index_stock_count_url,
    zh_sina_index_stock_hist_url,
)
from akshare.stock.cons import hk_js_decode
from akshare.utils import demjson

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] =False #减号unicode编码

st.set_page_config(page_icon="🎇",)
st.markdown("# 相关性分析")
st.sidebar.header("相关性分析")
CIindustryindex_list = [
    'CI005030.WI', 'CI005029.WI', 'CI005028.WI', 'CI005027.WI', 'CI005026.WI', 'CI005025.WI', 'CI005024.WI',
    'CI005023.WI', 'CI005022.WI', 'CI005021.WI', 'CI005020.WI', 'CI005019.WI', 'CI005018.WI', 'CI005017.WI',
    'CI005016.WI', 'CI005015.WI', 'CI005014.WI', 'CI005013.WI', 'CI005012.WI', 'CI005011.WI', 'CI005010.WI',
    'CI005009.WI', 'CI005008.WI', 'CI005007.WI', 'CI005006.WI', 'CI005005.WI', 'CI005004.WI', 'CI005003.WI',
    'CI005002.WI', 'CI005001.WI']

CIStyleindex_list = ['CI005921.WI', 'CI005920.WI', 'CI005919.WI', 'CI005918.WI', 'CI005917.WI']

Styleindex_list = ['399372.SZ', '399373.SZ', '399374.SZ', '399375.SZ', '399376.SZ', '399377.SZ']
Stockindex_list = ['000300.SH', '000905.SH', '000852.SH', '881001.WI']

benchlist = CIindustryindex_list+CIStyleindex_list+Styleindex_list+Stockindex_list

dict_1={"中盘成长":"sz399374","中盘价值":"sz399375",
 "小盘成长":"sz399376","小盘价值":"sz399377",
 "大盘成长":"sz399372","大盘价值":"sz399373"}

def stock_zh_index_daily(name: str = "sh000922") -> pd.DataFrame:
    """
    新浪财经-指数-历史行情数据, 大量抓取容易封 IP
    https://finance.sina.com.cn/realstock/company/sh000909/nc.shtml
    :param symbol: sz399998, 指定指数代码
    :type symbol: str
    :return: 历史行情数据
    :rtype: pandas.DataFrame
    """
    symbol=dict_1[name]
    params = {"d": "2020_2_4"}
    res = requests.get(zh_sina_index_stock_hist_url.format(symbol), params=params)
    js_code = py_mini_racer.MiniRacer()
    js_code.eval(hk_js_decode)
    dict_list = js_code.call(
        "d", res.text.split("=")[1].split(";")[0].replace('"', "")
    )  # 执行js解密代码
    temp_df = pd.DataFrame(dict_list)
    temp_df["date"] = pd.to_datetime(temp_df["date"]).dt.date
    temp_df["open"] = pd.to_numeric(temp_df["open"])
    temp_df["close"] = pd.to_numeric(temp_df["close"])
    temp_df["high"] = pd.to_numeric(temp_df["high"])
    temp_df["low"] = pd.to_numeric(temp_df["low"])
    temp_df["volume"] = pd.to_numeric(temp_df["volume"])
    df=temp_df[['date','close']]
    df=df.rename(columns={'close':f'{name}'})
    return df

#缓存指数数据
@st.cache_data
def load_data():
    df1=stock_zh_index_daily(name="中盘成长")
    for i in ["中盘价值","小盘成长","小盘价值","大盘成长","大盘价值"]:
        df=stock_zh_index_daily(name=i)
        df1=pd.merge(df1,df,on='date')
    df1['date']=pd.to_datetime(df1['date'])
    df_hist1=df1.rename(columns={'date':'tradedate'})
    df_hist1['tradedate']=df_hist1['tradedate'].apply(lambda x:str(x)[:4]+str(x)[5:7]+str(x)[8:])
    df_hist1['tradedate']=df_hist1['tradedate'].apply(lambda x:x[:9])
    return df_hist1
df_hist2=load_data()

@st.cache_data
def load_index():
    df=pd.read_csv("指数数据.csv")
    df['tradedate']=pd.to_datetime(df['tradedate'])
    df['tradedate']=df['tradedate'].apply(lambda x:str(x)[:4]+str(x)[5:7]+str(x)[8:])
    df['tradedate']=df['tradedate'].apply(lambda x:x[:9])
    return df
df_hist3=load_index()

start_date = st.date_input(
    "请选择开始日期",
    date(2020,2,9))
#st.write('开始日期:', start_date)
开始=str(start_date)
end_date = st.date_input(
    "请选择结束日期",
    date(2021,5,9))
#st.write('结束日期:',end_date)
结束=str(end_date)

code=st.text_input('请输入基金代码例如000001.OF')

wind_index=df_hist3[['tradedate']+Stockindex_list]

wind_index['tradedate']=pd.to_datetime(wind_index['tradedate'])
df_hist2['tradedate']=pd.to_datetime(df_hist2['tradedate'])

wind_index.set_index('tradedate', inplace=True)
wind_index_part = wind_index[开始:结束]  # 区间参数

style_index=df_hist2.set_index('tradedate')
style_index_lf=style_index[开始:结束]

if code:
    df = pro.fund_nav(ts_code=code,start_date=开始,end_date=结束,fields=['ts_code','nav_date','accum_nav'])
    #df['nav_date']=pd.to_datetime(df['nav_date'])
    df1=df.sort_values(by='nav_date',ignore_index=True)
    df9=df1[['nav_date','accum_nav']]
    df2=df9.rename(columns={'nav_date':'tradedate','accum_nav':code})
    df2['tradedate']=pd.to_datetime(df2['tradedate'])
    df3=df2.set_index('tradedate')
    df4=df3[开始:结束]
    nv=pd.merge(wind_index_part,df4,left_index=True,right_index=True)#将宽基指数数据和基金净值数据合并
    style_nav=pd.merge(style_index_lf,df4,left_index=True,right_index=True)

    corr1=nv.corr()
    fig=plt.figure(figsize=(10,4))
    sns.heatmap(corr1,annot=True)
    st.pyplot(fig)

    corr2=style_nav.corr()
    fig1=plt.figure(figsize=(10,4))
    sns.heatmap(corr2,annot=True)
    st.pyplot(fig1)

    

