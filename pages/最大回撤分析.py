import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta,date
import plotly as py
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import tushare as ts
pro = ts.pro_api('8e812052c92d7a829f0e3b0197d248e48bb2ba3efbbaa60f505e6852')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] =False #减号unicode编码
st.set_page_config(page_icon="🎇",)
st.markdown("# 最大回撤分析")
st.sidebar.header("最大回撤分析")
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
if code:
    df = pro.fund_nav(ts_code=code,start_date=开始,end_date=结束,fields=['ts_code','nav_date','accum_nav'])
    df8=df.sort_values(by='nav_date',ignore_index=True)
    df2=df8.rename(columns={'nav_date':'time'})
    d=df2[['time','accum_nav']]
    d['time']=pd.to_datetime(d['time'])
    dp=d.set_index('time')
    dp['drawdown']=-((dp['accum_nav'].cummax() - dp['accum_nav']) /
                                        (dp['accum_nav'].cummax()))

    dp['drawdown']=dp['drawdown'].astype('float64')
    if len(dp['drawdown'])>2:
        min_dd=dp['drawdown'].idxmin()#最低点
        
        qian=dp.loc[:min_dd]#前区间
        max_time=qian['accum_nav'].cummax().idxmax()
        最大回撤形成时间=min_dd-max_time
        hou=dp.loc[min_dd:]#后区间
        后区间的累计最大值=hou['accum_nav'].cummax().max()
        前区间的累计最大值=qian['accum_nav'].cummax().max()   
        if 后区间的累计最大值>=前区间的累计最大值:
            sd=hou.loc[hou['accum_nav']>=前区间的累计最大值]
            z=sd.iloc[0].name#大于等于前区间累计最大值的第一个日期
            修复天数=z-min_dd
            st.write(f"{code}基金最大回撤形成天数为{int(最大回撤形成时间.days)}天,最大回撤修复天数为{int(修复天数.days)}天")
        else:
            st.write(f"{code}基金最大回撤形成天数为{int(最大回撤形成时间.days)}天，最大回撤尚未修复")
    dp=dp.reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dp['time'],
        y=dp['drawdown'],
        fill='tozeroy',
        name='累计单位净值回撤',
        xaxis='x2',
        yaxis='y2'))
    fig.update_layout(
        title_text=code + "回撤情况",
        )
    st.plotly_chart(fig)