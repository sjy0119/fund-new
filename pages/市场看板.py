import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import akshare as ak
from plotly.figure_factory import create_table
import asyncio
import aiohttp
import requests
import json
from akshare.utils import demjson
from plotly.subplots import make_subplots
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] =False 

st.title('市场看板 :blue[!] :sunglasses:')
st.markdown('1.市场指数情况')
st.markdown('2.股票市场情况')
st.markdown('3.风格因子情况')
st.markdown('4.宏观指标情况及私募策略指数情况')

#爬虫函数定义模块
#1.爬取市场主要指数的收盘价数据
global_index_list=[f'http://push2his.eastmoney.com/api/qt/stock/kline/get?secid={i}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&lmt=58&klt=101&fqt=1&beg=20070115&end=20500101&ut=4f1862fc3b5e77c150a2b985b12db0fd&cb=cb_1699079114473_83662397&cb_1699079114473_83662397=cb_1699079114473_83662397'
                  for i in ['1.000300','1.000905','1.000906','1.000852','1.000016','1.000688']]

global_name=['沪深300','中证500','上证50','中证1000','中证800','科创50']

@st.cache_data(ttl=60)
def get_data():
    data=[]
    for url , i in zip(global_index_list,global_name):
        r=requests.get(url)
        data_text =  r.text
        df=eval(data_text[data_text.find("(") + 1 :-2])
        temp_df=pd.DataFrame([item.split(",") for item in df["data"]["klines"]]).iloc[:,:5]
        temp_df.columns = ["date", "open", "close", "high", "low"]
        temp_df=temp_df[['date','close']]
        temp_df['date']=pd.to_datetime(temp_df['date'])
        temp_df["close"] = pd.to_numeric(temp_df["close"], errors="coerce")
        temp_df=temp_df.rename(columns={'close':i})
        temp_df=temp_df.set_index('date')
        data.append(temp_df)
    global_index_df=pd.concat(data,axis=1).fillna(method='pad',axis=0)
    return global_index_df
#所有指数数据
all_data=get_data()
#2.读取barra因子收益率数据
@st.cache_data(ttl=6000)
def load_barra_data():
    barra_factor=pd.read_csv("factor_return.csv",index_col=0)#读取barra因子日收益率数据
    barra_factor=barra_factor.rename(columns={'price_date':"date"})
    barra_factor['date']=pd.to_datetime(barra_factor['date'])
    barra_factor1=barra_factor.set_index('date')
    barra_factor1.columns=['BarraCNE5_Beta', 'BarraCNE5_BooktoPrice', 'BarraCNE5_DebttoAssets',
    'BarraCNE5_EarningsYield', 'BarraCNE5_Growth', 'BarraCNE5_Liquidity',
    'BarraCNE5_Momentum', 'BarraCNE5_NonLinearSize',
    'BarraCNE5_ResidualVolatility', 'BarraCNE5_Size']
    for barra_col in barra_factor1.columns:
        barra_factor1[barra_col+'_nav'] = 1
        barra_factor1[barra_col+'_nav'].iloc[1:] = (1+barra_factor1[barra_col].iloc[1:]).cumprod()
    barra_factor2_list =['BarraCNE5_Beta_nav', 'BarraCNE5_BooktoPrice_nav', 'BarraCNE5_DebttoAssets_nav',
    'BarraCNE5_EarningsYield_nav', 'BarraCNE5_Growth_nav', 'BarraCNE5_Liquidity_nav',
    'BarraCNE5_Momentum_nav', 'BarraCNE5_NonLinearSize_nav',
    'BarraCNE5_ResidualVolatility_nav', 'BarraCNE5_Size_nav']
    for i in barra_factor2_list:
        barra_factor1[i+'_pct'] = barra_factor1[i].pct_change()
    return barra_factor1
#barra收益率
barra_factor=load_barra_data()
#名称列表
barra_factor_list =['BarraCNE5_Beta_nav_pct', 'BarraCNE5_BooktoPrice_nav_pct', 'BarraCNE5_DebttoAssets_nav_pct',
    'BarraCNE5_EarningsYield_nav_pct', 'BarraCNE5_Growth_nav_pct', 'BarraCNE5_Liquidity_nav_pct',
    'BarraCNE5_Momentum_nav_pct', 'BarraCNE5_NonLinearSize_nav_pct',
    'BarraCNE5_ResidualVolatility_nav_pct', 'BarraCNE5_Size_nav_pct']
#3.读取期货风格因子收益率数据
@st.cache_data
def load_future_data():
    df=pd.read_csv("期货风格因子日收益率",index_col=0)
    df=df.rename(columns={'price_date':'date'})
    df=df.set_index('date')
    return df
future_data=load_future_data()

def cai_future_return(future_data):
    future_data1=future_data.copy()
    dfff=future_data1.iloc[:,:16].iloc[-22:,:]
    for i in dfff.columns:
        dfff[i+'nav']=1
        dfff[i+'nav'].iloc[1:]=(1+dfff[i].iloc[1:]).cumprod()
        dfff[i+'nav']=dfff[i+'nav']-1
    return dfff.iloc[:,16:]
future_data_return=cai_future_return(future_data)

#4.计算barra因子暴露度
def barra_ana(df,code):
    lf=df.copy()
    b1 = np.array(lf[code])  # 因变量
    A1 = np.array(lf[barra_factor_list])
    def minmean(A1,b1):
        num_x = np.shape(A1)[1]
        def my_func(x):
            ls = np.abs((b1-np.dot(A1,x))**2)
            #ld=lam*np.sum([pow(x[n], 2) for n in range(7, 17)])
            result = np.sum(ls)#+ld
            return result
        def g1(x):
            return np.sum(x) #sum of X >= 0
        def g2(x):
            return 1-np.sum(x) #sum of X = 1
        cons = ({'type': 'ineq', 'fun': g1}
                ,{'type': 'eq', 'fun':  g2})
        x0  = np.array([ -100, -100, -100, -
                100, -100, -100, -100, -100, -100, -100])
        bnds = [ (None, None), (None, None), (None, None), (None, None), (None, None),
            (None, None), (None, None), (None, None), (None, None), (None, None)]
        res = minimize(my_func, 
                    bounds = bnds, x0=x0,
                    constraints=cons)
        return res
    res=minmean(A1,b1)
    beta_1 = res.x[0]  
    beta_2 = res.x[1]
    beta_3 = res.x[2]
    beta_4 = res.x[3]
    beta_5 = res.x[4]
    beta_6 = res.x[5]
    beta_7 = res.x[6]
    beta_8 = res.x[7]
    beta_9 = res.x[8]
    beta_10 = res.x[9]
    
    CIS_Barra =dict()
    CIS_Barra['Beta因子暴露'] = beta_1
    CIS_Barra['账面市值比因子暴露'] = beta_2
    CIS_Barra['盈利预期因子暴露'] = beta_3
    CIS_Barra['成长因子暴露'] = beta_4
    CIS_Barra['杠杆因子暴露'] = beta_5
    CIS_Barra['流动性因子暴露'] = beta_6
    CIS_Barra['动量因子暴露'] = beta_7
    CIS_Barra['非线性市值因子暴露'] = beta_8
    CIS_Barra['残差波动率因子暴露'] = beta_9
    CIS_Barra['市值因子暴露'] = beta_10
    CIS_Barra=pd.DataFrame([CIS_Barra])
    return CIS_Barra,res
#5.读取申万风格指数数据
@st.cache_data(ttl=6000)
def load_sw():
    df=pd.read_csv("sw风格指数数据",index_col=0)
    df.index=pd.DatetimeIndex(df.index)
    return df
sw_style=load_sw()
#6.读取申万以及行业数据
@st.cache_data(ttl=6000)
def load_sw_1():
    df=pd.read_csv("申万一级数据",index_col=0)
    list1=list(df['指数名称'].unique())
    df_list=[df.loc[df['指数名称']==i][['日期','收盘价']].rename(columns={'收盘价':i}).set_index('日期') for i in list1]
    df_all=pd.concat(df_list,axis=1)
    df_all.index=pd.DatetimeIndex(df_all.index)
    return df_all
sw_1=load_sw_1()
#7.读取私募策略指数数据
@st.cache_data
def load_simu_index():
    df=pd.read_csv("私募指数数据",index_col=0)
    list1=list(df['火富牛私募策略指数'].unique())
    df_list=[df.loc[df['火富牛私募策略指数']==i][['price_date','nav']].rename(columns={'nav':i}).set_index('price_date') for i in list1]
    df_all=pd.concat(df_list,axis=1)
    df_all=df_all.reset_index().sort_values(by='price_date').set_index('price_date')
    return df_all
simu_index=load_simu_index()
#获取概念板块主力近5日资金流入
@st.cache_data(ttl=60)
def get_tech_data():
    url='https://push2.eastmoney.com/api/qt/clist/get?cb=jQuery1123007130534607061079_1699333084192&fid=f164&po=1&pz=500&pn=1&np=1&fltt=2&invt=2&ut=b2884a393a59ad64002292a3e90d46a5&fs=m%3A90+t%3A3&fields=f12%2Cf14%2Cf2%2Cf109%2Cf164%2Cf165%2Cf166%2Cf167%2Cf168%2Cf169%2Cf170%2Cf171%2Cf172%2Cf173%2Cf257%2Cf258%2Cf124%2Cf1%2Cf13'
    r=requests.get(url)
    data_text=r.text
    data=data_text[43:-2]
    df=pd.DataFrame(eval(data)['data']['diff'])
    df1=df.loc[:,['f14','f164']]
    df1.columns=['概念名称','主力净流入']
    df1['主力净流入']=df1['主力净流入'].apply(lambda x: round(x/100000000,2))
    return df1
tech_data=get_tech_data()

#获取大盘资金流向
@st.cache_data(ttl=600)
def get_money_flow():
    url='https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get?cb=jQuery11230960759451624605_1699334564622&lmt=0&klt=101&fields1=f1%2Cf2%2Cf3%2Cf7&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61%2Cf62%2Cf63%2Cf64%2Cf65&ut=b2884a393a59ad64002292a3e90d46a5&secid=1.000001&secid2=0.399001&_=1699334564623'
    r=requests.get(url)
    data_text=r.text
    data_=pd.DataFrame(items.split(',') for items in eval(data_text[41:-2])['data']['klines'])
    data_=data_.iloc[:,:6]
    data_.columns=['date','主力净流入','小单净流入','中单净流入','大单净流入','超大单净流入']
    for i in data_.columns[1:]:
        data_[i]=data_[i].apply(lambda x: round(float(x)/100000000.0,3))
    return data_
money=get_money_flow()
money['date']=pd.to_datetime(money['date'])
money=money.set_index('date')
#获取板块今日资金净流入情况
@st.cache_data(ttl=60)
def get_industry():
    url='https://push2.eastmoney.com/api/qt/clist/get?cb=jQuery112305790360726884456_1699335338708&pn=1&pz=500&po=1&np=1&fields=f12%2Cf13%2Cf14%2Cf62&fid=f62&fs=m%3A90%2Bt%3A2&ut=b2884a393a59ad64002292a3e90d46a5&_=1699335338729'
    r=requests.get(url)
    data_text=r.text
    data=pd.DataFrame(eval(data_text[42:-2])['data']['diff'])
    data=data.iloc[:,2:]
    data.columns=['行业名称','资金净流入']
    data['资金净流入']=data['资金净流入'].apply(lambda x: round(float(x)/100000000,3))
    return data
industry_money=get_industry()

#获取股票主力排名
st.cache_data(ttl=60)
def best_data():
    url='https://push2.eastmoney.com/api/qt/clist/get?cb=jQuery112309128099157586371_1699335863813&fid=f184&po=1&pz=50&pn=1&np=1&fltt=2&invt=2&fields=f2%2Cf3%2Cf12%2Cf13%2Cf14%2Cf62%2Cf184%2Cf225%2Cf165%2Cf263%2Cf109%2Cf175%2Cf264%2Cf160%2Cf100%2Cf124%2Cf265%2Cf1&ut=b2884a393a59ad64002292a3e90d46a5&fs=m%3A0%2Bt%3A6%2Bf%3A!2%2Cm%3A0%2Bt%3A13%2Bf%3A!2%2Cm%3A0%2Bt%3A80%2Bf%3A!2%2Cm%3A1%2Bt%3A2%2Bf%3A!2%2Cm%3A1%2Bt%3A23%2Bf%3A!2%2Cm%3A0%2Bt%3A7%2Bf%3A!2%2Cm%3A1%2Bt%3A3%2Bf%3A!2'
    r=requests.get(url)
    data_text=r.text
    df=pd.DataFrame(json.loads(data_text[42:-2])['data']['diff']).iloc[:20,:]
    df1=df[['f12','f14','f100','f2','f225','f263','f264']]
    df1.columns=['股票代码','股票名称','行业','最新价','今日排名','5日排名','10日排名']
    df1.loc[:,'最新价']=df1.loc[:,'最新价'].apply(lambda x: round(x,2))
    return df1
best_stock_data=best_data()

#北向资金情况
@st.cache_data(ttl=600)
def north_money():
    url='https://push2his.eastmoney.com/api/qt/kamt.kline/get?fields1=f1,f3,f5&fields2=f51,f52&klt=101&lmt=500&ut=b2884a393a59ad64002292a3e90d46a5&cb=jQuery112309386865849321027_1699336618719&_=1699336618731'
    r=requests.get(url)
    data_text=r.text
    data=pd.DataFrame([items.split(',') for items in json.loads(data_text[42:-2])['data']['s2n']])
    data.columns=['date','北向资金净流入']
    data.loc[:,'北向资金净流入']=data.loc[:,'北向资金净流入'].apply(lambda x: round(float(x)/10000,2))
    return data
north_money1=north_money()
#CPI数据
@st.cache_data(ttl=6000)
def chinese_cpi_index():
    url='https://datacenter-web.eastmoney.com/api/data/v1/get?callback=datatable1798191&columns=REPORT_DATE%2CTIME%2CNATIONAL_SAME%2CNATIONAL_BASE%2CNATIONAL_SEQUENTIAL%2CNATIONAL_ACCUMULATE%2CCITY_SAME%2CCITY_BASE%2CCITY_SEQUENTIAL%2CCITY_ACCUMULATE%2CRURAL_SAME%2CRURAL_BASE%2CRURAL_SEQUENTIAL%2CRURAL_ACCUMULATE&pageNumber=1&pageSize=50&sortColumns=REPORT_DATE&sortTypes=-1&source=WEB&client=WEB&reportName=RPT_ECONOMY_CPI&p=1&pageNo=1&pageNum=1&_=1699337739366'
    r=requests.get(url)
    data_text=r.text
    df=pd.DataFrame(json.loads(data_text[17:-2])['result']['data'])
    df1=df[['REPORT_DATE','NATIONAL_BASE','CITY_BASE','RURAL_BASE']]
    df1.columns=['date','全国','城市','农村']
    df1.loc[:,'date']=pd.to_datetime(df1.loc[:,'date'])
    df1=df1.sort_values(by='date')
    df1=df1.set_index('date')
    return df1
cpi=chinese_cpi_index()
#PPI
@st.cache_data(ttl=6000)
def get_ppi():
    url='https://datacenter-web.eastmoney.com/api/data/v1/get?callback=datatable6955761&columns=REPORT_DATE%2CTIME%2CBASE%2CBASE_SAME%2CBASE_ACCUMULATE&pageNumber=1&pageSize=50&sortColumns=REPORT_DATE&sortTypes=-1&source=WEB&client=WEB&reportName=RPT_ECONOMY_PPI&p=1&pageNo=1&pageNum=1&_=1699338734836'
    r=requests.get(url)
    data_text=r.text
    df=pd.DataFrame(json.loads(data_text[17:-2])['result']['data'])
    df=df[['REPORT_DATE','BASE','BASE_ACCUMULATE']]
    df.columns=['date','当月','累计']
    df.loc[:,'date']=pd.to_datetime(df.loc[:,'date'])
    df=df.sort_values(by='date')
    df=df.set_index('date')
    return df
ppi=get_ppi()
#PMI
@st.cache_data(ttl=6000)
def get_pmi():
    url='https://datacenter-web.eastmoney.com/api/data/v1/get?callback=datatable5700962&columns=REPORT_DATE%2CTIME%2CMAKE_INDEX%2CMAKE_SAME%2CNMAKE_INDEX%2CNMAKE_SAME&pageNumber=1&pageSize=100&sortColumns=REPORT_DATE&sortTypes=-1&source=WEB&client=WEB&reportName=RPT_ECONOMY_PMI&p=1&pageNo=1&pageNum=1&_=1699339188452'
    r=requests.get(url)
    data_text=r.text
    df=pd.DataFrame(json.loads(data_text[17:-2])['result']['data'])
    df=df[['REPORT_DATE','MAKE_INDEX','NMAKE_INDEX']]
    df.columns=['date','制造业','非制造业']
    df.loc[:,'date']=pd.to_datetime(df.loc[:,'date'])
    df=df.sort_values(by='date').set_index('date')
    return df
pmi=get_pmi()
#上海间隔夜拆借利率
@st.cache_data(ttl=6000)
def get_shibor():
    r=requests.get('https://datacenter-web.eastmoney.com/api/data/v1/get?callback=datatable8889269&reportName=RPT_IMP_INTRESTRATEN&columns=REPORT_DATE%2CREPORT_PERIOD%2CIR_RATE%2CCHANGE_RATE%2CINDICATOR_ID%2CLATEST_RECORD%2CMARKET%2CMARKET_CODE%2CCURRENCY%2CCURRENCY_CODE&quoteColumns=&filter=(LATEST_RECORD%3D1)(MARKET_CODE%3D%22001%22)&pageNumber=1&pageSize=20&sortTypes=1&sortColumns=INDICATOR_ID&source=WEB&client=WEB&_=1699339690131')
    data_text=r.text
    df=pd.DataFrame(json.loads(data_text[17:-2])['result']['data'])
    df=df[['REPORT_DATE','REPORT_PERIOD','IR_RATE','CHANGE_RATE','MARKET']]
    df.columns=['date','品种','利率(%)','涨跌(BP)','市场']
    df['date']=pd.to_datetime(df['date'])
    df['date']=df['date'].apply(lambda x:x.strftime('%Y-%m-%d'))
    return df
shibor=get_shibor()
#定义一个计算barra暴露度的函数
all_data1=all_data.pct_change().fillna(0)
df_all=pd.merge(barra_factor,all_data1,left_index=True,right_index=True,how='left')
#计算指数在barra上的暴露度
def cal_barra(df_all):
    all1=df_all.copy()
    all1=all1.iloc[-252:,]
    ddd=pd.concat([ barra_ana(all1,i)[0] for i in ['沪深300','中证500','中证1000','中证800','上证50','科创50']])
    ddd.insert(0,'指数名称',['沪深300','中证500','中证1000','中证800','上证50','科创50'])
    for i in ddd.columns[1:]:
      ddd[i]=ddd[i].apply(lambda x: round(x,3))
    D=ddd.T
    return D

exposure=cal_barra(df_all)

def cal_barra_return(barra_factor):
    barra_factor1=barra_factor.copy()
    dfff=barra_factor1.iloc[:,:10].iloc[-22:,:]
    for i in dfff.columns:
        dfff[i+'nav']=1
        dfff[i+'nav'].iloc[1:]=(1+dfff[i].iloc[1:]).cumprod()
        dfff[i+'nav']=dfff[i+'nav']-1
    return dfff.iloc[:,10:]

barra_return=cal_barra_return(barra_factor)

#指数的周度涨跌幅
chinese_index_name_list=list(all_data.columns)
week_return=[round(i*100,2) for i in all_data.iloc[-1,]/all_data.iloc[-5,]-1]
#计算申万风格指数的周度涨跌幅
sw_name=list(sw_style.columns)
sw_return=[round(i*100,2) for i in list(sw_style.iloc[-1,:]/sw_style.iloc[-5,:]-1)]
#计算申万一级行业的月度涨跌幅
sw_1_name=list(sw_1.columns)
sw_1_return=[round(i*100,2) for i in list(sw_1.iloc[-1,:]/sw_1.iloc[-22,:]-1)]
#申万一级行业近一年的
sw_1_return1=[round(i*100,2) for i in list(sw_1.iloc[-1,:]/sw_1.iloc[-252,:]-1)]

def RV(x):
    return np.sqrt(np.sum(x**2))

def cai_std(all_data):
    all_data2=all_data.pct_change().fillna(0)
    index1=all_data2.groupby(pd.Grouper(freq='5d')).apply(RV).iloc[-22:,]
    index2=all_data2.groupby(pd.Grouper(freq='10d')).apply(RV).iloc[-22:,]
    index3=all_data2.groupby(pd.Grouper(freq='20d')).apply(RV).iloc[-22:,]

    return index1,index2,index3

index=cai_std(all_data)

with st.container():
   fig = go.Figure(data = (
        go.Bar(x=chinese_index_name_list,  # x轴数据
               y=week_return,text=week_return,textposition="outside" # y轴数据
              )))
   fig.update_layout(title_text='主要指数的周度涨跌幅:'+all_data.index[-1].strftime('%Y-%m-%d')) 
   st.plotly_chart(fig)

   fi8 = go.Figure()
   d1=[]
   for i in index[0].columns:
       d1.append(go.Scatter(x=index[0].index.strftime('%Y-%m-%d'),y=index[0][i],mode='lines',name=i))
   fi8 = go.Figure(data=d1)

# 柱状图模式需要设置：4选1
   fi8.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5)) 
   st.markdown('**各指数5日时序波动率**')
   st.plotly_chart(fi8)

   fi9 = go.Figure()
   d2=[]
   for i in index[1].columns:
       d2.append(go.Scatter(x=index[1].index.strftime('%Y-%m-%d'),y=index[1][i],mode='lines',name=i))
   fi9 = go.Figure(data=d2)

# 柱状图模式需要设置：4选1
   fi9.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5)) 
   st.markdown('**各指数10日时序波动率**')
   st.plotly_chart(fi9)

   fi10 = go.Figure()
   d3=[]
   for i in index[2].columns:
       d3.append(go.Scatter(x=index[2].index.strftime('%Y-%m-%d'),y=index[2][i],mode='lines',name=i))
   fi10 = go.Figure(data=d3)

# 柱状图模式需要设置：4选1
   fi10.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5)) 
   st.markdown('**各指数20日时序波动率**')
   st.plotly_chart(fi10)


   year_data=all_data.iloc[-252:,:]
   year_data=year_data/year_data.iloc[0,]
   sw_year_data=sw_style.iloc[-252:,:]
   sw_year_data=sw_year_data/sw_year_data.iloc[0,]
   sw_1_year_data=sw_1.iloc[-252:,:]
   sw_1_year_data=sw_1_year_data/sw_1_year_data.iloc[0,]
   #绘制近一年的收盘价
   lines=[]
   for i in year_data.columns:
       lines.append(go.Scatter(x=year_data.index.strftime('%Y-%m-%d'),y=list(year_data[i]),name=i))
   fig4 = go.Figure(data=lines)
   fig4.update_layout(title_text='指数近一年的收盘价走势(归一化处理):'+year_data.index[-1].strftime('%Y-%m-%d'),legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5))
   fig4.update_xaxes(rangeslider_visible=True,linecolor='#c5c5c5')
   st.plotly_chart(fig4)

   fig1 = go.Figure(data = (
        go.Bar(x=sw_name,  # x轴数据
               y=sw_return,text=week_return,textposition="outside" # y轴数据
              )))
   fig1.update_layout(title_text='申万风格指数的周度涨跌幅:'+sw_style.index[-1].strftime('%Y-%m-%d')) 
   st.plotly_chart(fig1)
   lines1=[]
   for i in sw_year_data.columns:
       lines1.append(go.Scatter(x=sw_year_data.index.strftime('%Y-%m-%d'),y=list(sw_year_data[i]),name=i))
   fig5 = go.Figure(data=lines1)
   fig5.update_layout(title_text='申万风格指数近一年的收盘价走势(归一化处理):'+sw_year_data.index[-1].strftime('%Y-%m-%d'),legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5))
   fig5.update_xaxes(rangeslider_visible=True,linecolor='#c5c5c5')
   st.plotly_chart(fig5)

   fig2 = go.Figure(data = (
        go.Bar(x=sw_1_name,  # x轴数据
               y=sw_1_return,text=sw_1_return,textposition="outside" # y轴数据
              )))
   fig2.update_layout(title_text='申万一级行业指数的近一月涨跌幅:'+sw_1.index[-1].strftime('%Y-%m-%d')) 
   st.plotly_chart(fig2)

   fig3 = go.Figure(data = (
        go.Bar(x=sw_1_name,  # x轴数据
               y=sw_1_return1,text=sw_1_return1,textposition="outside" # y轴数据
              )))
   fig3.update_layout(title_text='申万一级行业指数的近一年涨跌幅:'+sw_1.index[-1].strftime('%Y-%m-%d')) 
   st.plotly_chart(fig3)

   lines2=[]
   for i in sw_1_year_data.columns:
       lines2.append(go.Scatter(x=sw_1_year_data.index.strftime('%Y-%m-%d'),y=list(sw_1_year_data[i]),name=i))
   fig6 = go.Figure(data=lines2)
   fig6.update_layout(legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5))
   fig6.update_xaxes(rangeslider_visible=True,linecolor='#c5c5c5')
   st.plotly_chart(fig6)


   fi= go.Figure(data = (
        go.Bar(x=list(tech_data['概念名称'].iloc[:10])+list(tech_data['概念名称'].iloc[-10:]),  # x轴数据
                y=list(tech_data['主力净流入'].iloc[:10])+list(tech_data['主力净流入'].iloc[-10:]),
                text=list(tech_data['主力净流入'].iloc[:10])+list(tech_data['主力净流入'].iloc[-10:]),
                textposition="outside" ,marker_color='#af0010'
                )))
   fi.update_layout(title_text='近5日概念主力净流入状况前10+后10,单位(亿)') 
   st.plotly_chart(fi)
   
   fi2= go.Figure(data = (
        go.Bar(x=list(industry_money['行业名称'].iloc[:10])+list(industry_money['行业名称'].iloc[-10:]),  # x轴数据
                y=list(industry_money['资金净流入'].iloc[:10])+list(industry_money['资金净流入'].iloc[-10:]),
                text=list(industry_money['资金净流入'].iloc[:10])+list(industry_money['资金净流入'].iloc[-10:]),
                textposition="outside" ,marker_color='#af0010'
                )))
   fi2.update_layout(title_text='今日日行业主力净流入状况前10+后10,单位(亿)') 
   st.plotly_chart(fi2)

   
   lines3=[]
   for i in money.columns:
            
        lines3.append(go.Scatter(x=money.index.strftime('%Y-%m-%d'),y=list(money[i]),name=i))
   fi1 = go.Figure(data=lines3)
   fi1.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5),title_text='资金流入状况:'+money.index[-1].strftime('%Y-%m-%d'))
   fi1.update_xaxes(rangeslider_visible=True,linecolor='#c5c5c5')
   st.plotly_chart(fi1)
   
   north_money1['date']=pd.to_datetime(north_money1['date'])
   north_money1=north_money1.set_index('date')
   fi3=go.Figure(data=[go.Scatter(x=north_money1.index.strftime('%Y-%m-%d'),y=list(north_money1['北向资金净流入']),name='北向资金净流入')])
   fi3.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5),title_text='北向资金流入状况:'+north_money1.index[-1].strftime('%Y-%m-%d')+'单位(亿元)')
   fi3.update_xaxes(rangeslider_visible=True,linecolor='#c5c5c5')
   st.plotly_chart(fi3)
   
   st.markdown('**今日股票排名**:'+north_money1.index[-1].strftime('%Y-%m-%d'))
   table=create_table(best_stock_data)
   st.plotly_chart(table)

   categories = ['Beta因子暴露',
    '账面市值比因子暴露',
    '盈利预期因子暴露',
    '成长因子暴露',
    '杠杆因子暴露',
    '流动性因子暴露',
    '动量因子暴露',
    '非线性市值因子暴露',
    '残差波动率因子暴露',
    '市值因子暴露'] 
   data__=[]
   for i,j in zip(['沪深300','中证500','中证1000','中证800','上证50','科创50'],range(7)):
    data__.append(go.Scatterpolar(name=i,theta=categories,r=list(exposure.iloc[:,j].values[1:]),fill='toself'))
   fi41 = go.Figure(data=data__)
   fi41.update_layout(polar=dict(
                            radialaxis=dict(
                            visible=True
                            )),
                        title_text='指数barra因子暴露度') 
   st.plotly_chart(fi41) 

   all_d=list(all_data.iloc[-1,]/all_data.iloc[-252,]-1)
   all__=list(all_data.columns)

   exposure1=exposure.iloc[1:,]
   exposure1.columns=exposure.iloc[0,:]
   exposure1=exposure.iloc[1:,]
   exposure1.columns=exposure.iloc[0,:]
   exposure1=exposure1.reset_index().rename(columns={'index':'因子暴露度'})
   exposure1=exposure1.set_index('因子暴露度')

   for i,j in zip(all_d,all__):
       exposure1[j]=exposure1[j]*i
   exposure1=exposure1.applymap(lambda x: round(x*100,3))
   dgf=pd.DataFrame()
   dgf['因子暴露度']=[i[:-2] for i in categories]
   for i in all__:
       dgf[i]=list(exposure1[i])
   table1=create_table(dgf)
   st.markdown('近一年各指数在barra因子暴露上的收益率')
   st.plotly_chart(table1)
   
   categorie = ['Beta因子',
    '账面市值比因子',
    '盈利预期因子',
    '成长因子',
    '杠杆因子',
    '流动性因子',
    '动量因子',
    '非线性市值因子',
    '残差波动率因子',
    '市值因子'] 
   fi5 = go.Figure()
   data11=[]
   for i,j in zip(barra_return.columns,categorie):
    data11.append(go.Scatter(x=barra_return.index.strftime('%Y-%m-%d'),y=barra_return[i],mode='lines',name=j))
   fi5 = go.Figure(data=data11)

# 柱状图模式需要设置：4选1
   fi5.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5)) 
   st.markdown('**近一月以来barra因子的累计收益率**')
   st.plotly_chart(fi5)




   future_data_return.index=pd.DatetimeIndex(future_data_return.index)

   fi6 = go.Figure()
   data12=[]
   for i,j in zip(future_data_return.columns,list(future_data.columns)):
    data12.append(go.Scatter(x=future_data_return.index.strftime('%Y-%m-%d'),y=future_data_return[i],mode='lines',name=j))
   fi6 = go.Figure(data=data12)

# 柱状图模式需要设置：4选1
   fi6.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5)) 
   st.markdown('**近一月以来期货风格因子的累计收益率**')
   st.plotly_chart(fi6)

   fi13 = go.Figure()
   d13=[]
   for i in cpi.columns:
    d13.append(go.Scatter(x=cpi.index.strftime('%Y-%m-%d'),y=cpi[i],mode='lines',name=i))
   fi13 = go.Figure(data=d13)

# 柱状图模式需要设置：4选1
   fi13.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5)) 
   st.markdown('**CPI指数**')
   st.plotly_chart(fi13)

   fi14 = go.Figure()
   d14=[]
   for i in pmi.columns:
    d14.append(go.Scatter(x=pmi.index.strftime('%Y-%m-%d'),y=pmi[i],mode='lines',name=i))
   fi14 = go.Figure(data=d14)

# 柱状图模式需要设置：4选1
   fi14.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5)) 
   st.markdown('**PMI指数**')
   st.plotly_chart(fi14)

   fi15 = go.Figure()
   d15=[]
   for i in ppi.columns:
    d15.append(go.Scatter(x=ppi.index.strftime('%Y-%m-%d'),y=ppi[i],mode='lines',name=i))
   fi15 = go.Figure(data=d15)

# 柱状图模式需要设置：4选1
   fi15.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5)) 
   st.markdown('**PPI指数**')
   st.plotly_chart(fi15)


   table2=create_table(shibor)
   st.markdown('最新shibor')
   st.plotly_chart(table2)

   simu_index1=simu_index.iloc[-252:,]/simu_index.iloc[-252,]
   simu_index1.index=pd.DatetimeIndex(simu_index1.index)
   fi16 = go.Figure()
   d16=[]
   for i in simu_index1.columns:
    d16.append(go.Scatter(x=simu_index1.index.strftime('%Y-%m-%d'),y=simu_index1[i],mode='lines',name=i))
   fi16 = go.Figure(data=d16)

# 柱状图模式需要设置：4选1
   fi16.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5)) 
   st.markdown('**私募策略指数(归一化处理)**')
   st.plotly_chart(fi16)





