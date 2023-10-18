import streamlit as st
from datetime import datetime, time, timedelta,date
from time import strftime
from scipy.optimize import minimize
from statsmodels.regression.linear_model import OLS  # 线性回归
import plotly as py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio
import seaborn as sns
import tushare as ts
pro = ts.pro_api('8e812052c92d7a829f0e3b0197d248e48bb2ba3efbbaa60f505e6852')
import akshare as ak
#图片显示中文
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] =False #减号unicode编码
st.set_page_config(page_icon="😎",)
st.markdown("# 基金经理管理产品明细")
st.sidebar.header("基金经理管理产品明细")
name_str=st.text_input('请输入基金经理姓名')
if name_str:
    df = pro.fund_manager(name=name_str)
    df1=df[['ts_code','name','begin_date','nationality','end_date']]
    df3=df1[df1.isnull().T.any()] #无论哪列，有空值的被选出来
    #df3['begin_date']=pd.to_datetime(df3['begin_date'])
    #df3['code']=df3['ts_code'].apply(lambda x:x[:6])
    fund_bas=df3.loc[df3['nationality']=='中国']
    fund_fis= pro.fund_basic(ts_code=fund_bas['ts_code'].iloc[0],fields=["ts_code",
            "name",
            "management",
            "custodian",
            "fund_type",
            "found_date",
            "benchmark",
            "invest_type",
            "type"])
    fund_fis=fund_fis.rename(columns={'name':'简称','management':'管理人','custodian':'托管人','fund_type':'投资类型','found_date':'成立日期','benchmark':'业绩比较基准','invest_type':'投资风格','type':'基金类型'})
    fund_fis=fund_fis.set_index('ts_code').T
    for i in list(fund_bas['ts_code'].iloc[1:]):
        fund_info = pro.fund_basic(ts_code=i,fields=["ts_code",
            "name",
            "management",
            "custodian",
            "fund_type",
            "found_date",
            "benchmark",
            "invest_type",
            "type"])
        fund_info=fund_info.rename(columns={'name':'简称','management':'管理人','custodian':'托管人','fund_type':'投资类型','found_date':'成立日期','benchmark':'业绩比较基准','invest_type':'投资风格','type':'基金类型'})
        fund_another=fund_info.set_index('ts_code').T
        fund_fis=pd.merge(fund_fis,fund_another,left_index=True,right_index=True)
    
    
    @st.cache_data
    def fund_bn():
        fund_manager_df = ak.fund_manager(adjust='0')
        f=fund_manager_df[['姓名','累计从业时间','现任基金资产总规模','现任基金最佳回报']]
        f['累计从业时间']=f['累计从业时间'].apply(lambda x: str(x)+'天')
        f['现任基金资产总规模']=f['现任基金资产总规模'].apply(lambda x: str(x)+'亿')
        f['现任基金最佳回报']=f['现任基金最佳回报'].apply(lambda x: str(x)+'%')
        return f
    basic_info=fund_bn()
    info_fund=basic_info.loc[basic_info['姓名']==name_str]
    st.write('该基金经理累计从业天数'+info_fund['累计从业时间'].iloc[0],'现任基金资产总规模'+info_fund['现任基金资产总规模'].iloc[0],'现任基金最佳回报'+info_fund['现任基金最佳回报'].iloc[0])
    fund_line=[]
    return1=[]
    week=[]
    month=[]
    half_year=[]
    one_year=[]
    two_year=[]
    three_year=[]
    for i in range(len(fund_bas['ts_code'])):
        df = pro.fund_nav(ts_code=df3['ts_code'].iloc[i],start_date=str(df3['begin_date'].iloc[i]),fields=['ts_code','nav_date','accum_nav'])
        df['nav_date']=pd.to_datetime(df['nav_date'])
        fund_ret=df.sort_values(by='nav_date',ignore_index=True)
        fund_ret['ret']=fund_ret['accum_nav'].pct_change()
        fund_ret['cumulative']=(1+fund_ret['ret']).cumprod()-1
        return1.append( fund_ret['cumulative'].iloc[-1])
        month_ret=fund_ret['accum_nav'].iloc[-20:]
        month_info=month_ret.iloc[-1]/month_ret.iloc[0]-1
        month.append(month_info)
        week_ret=fund_ret['accum_nav'].iloc[-5:]
        week_info=week_ret.iloc[-1]/week_ret.iloc[0]-1
        week.append(week_info)
        if len(fund_ret['nav_date'])>=125:
            half_year_ret=fund_ret['accum_nav'].iloc[-125:]
            half_year_info=half_year_ret.iloc[-1]/half_year_ret.iloc[0]-1
            half_year.append(half_year_info)
        else:
            half_year.append(0)
        if len(fund_ret['nav_date'])>=250:
            one_year_ret=fund_ret['accum_nav'].iloc[-250:]
            one_year_info=one_year_ret.iloc[-1]/one_year_ret.iloc[0]-1
            one_year.append(one_year_info)
        else:
            one_year.append(0)
        if len(fund_ret['nav_date'])>=500:
            two_year_ret=fund_ret['accum_nav'].iloc[-500:]
            two_year_info=two_year_ret.iloc[-1]/two_year_ret.iloc[0]-1
            two_year.append(two_year_info)
        else:
            two_year.append(0)
        if len(fund_ret['nav_date'])>=750:
            three_year_ret=fund_ret['accum_nav'].iloc[-750:]
            three_year_info=three_year_ret.iloc[-1]/three_year_ret.iloc[0]-1
            three_year.append(three_year_info)
        else:
            three_year.append(0)
        line=go.Scatter(x=fund_ret['nav_date'],y=fund_ret['cumulative'], mode='lines', name=fund_bas['ts_code'].iloc[i])
        fund_line.append(line)
    fund_return=pd.DataFrame()
    fund_return['基金代码']=fund_bas['ts_code']
    fund_return['近一周收益率']=week
    fund_return['近一月收益率']=month
    fund_return['近半年收益率']=half_year
    fund_return['近一年收益率']=one_year
    fund_return['近两年收益率']=two_year
    fund_return['近三年收益率']=three_year
    fund_return['近一周收益率']=fund_return['近一周收益率'].apply(lambda x: format(x, '.2%'))
    fund_return['近一月收益率']=fund_return['近一月收益率'].apply(lambda x: format(x, '.2%'))
    fund_return['近半年收益率']=fund_return['近半年收益率'].apply(lambda x: format(x, '.2%'))
    fund_return['近一年收益率']=fund_return['近一年收益率'].apply(lambda x: format(x, '.2%'))
    fund_return['近两年收益率']=fund_return['近两年收益率'].apply(lambda x: format(x, '.2%'))
    fund_return['近三年收益率']=fund_return['近三年收益率'].apply(lambda x: format(x, '.2%'))
    fund_return=fund_return.set_index('基金代码')
    INF=fund_fis.T
    INF['任职期间收益']=return1
    INF['任职期间收益']=INF['任职期间收益'].apply(lambda x: format(x, '.2%'))
    st.dataframe(INF.T)
    if st.checkbox('展示该基金经理管理产品的累计收益情况'):
        fig_nav_CIS = go.Figure(data=fund_line)

        fig_nav_CIS .update_layout(
                            title_text="基金累计收益走势 <br> 最新净值日期:" ,
                            margin=dict(l=100, r=100, t=60, b=80),
                            yaxis={'tickformat': '.2f', 'title': ' 基金累计收益'},
                            xaxis={'tickangle': -70, 'tickformat': '%Y-%m-%d'})
        st.plotly_chart(fig_nav_CIS)
    if st.checkbox('展示该基金经理管理产品的分阶段收益情况'):
        st.subheader('如果为0则表示自管理日起尚未满该时间段')
        st.dataframe(fund_return)
