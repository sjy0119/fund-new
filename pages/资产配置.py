import pandas as pd
import numpy as np
import streamlit as st
from datetime import  date
from plotly.figure_factory import create_table
from scipy.optimize import minimize
from sklearn.covariance import ledoit_wolf
from matplotlib import pyplot as plt
import akshare as ak
import asyncio
import aiohttp
import plotly.graph_objs as go
from akshare.utils import demjson
import orjson
import requests
import warnings
import tushare as ts
pro = ts.pro_api('8e812052c92d7a829f0e3b0197d248e48bb2ba3efbbaa60f505e6852')
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] =False 

st.set_page_config(page_icon="😎",)

st.markdown("# 大类资产配置")
st.sidebar.header("大类资产配置")


index_list=['000300.SH','000905.SH','000906.SH','000852.SH','000016.SH','000688.SH']
global_list=['IXIC','SPX','HSI','N225','FTSE','GDAXI']
global_name=['纳斯达克','标普500','恒生指数','日经225','富时100','德国DAX30']
name_list=['沪深300','中证500','上证50','中证1000','中证800','科创50']
@st.cache_data(ttl=12000)
def get_data():
  all_df=pd.DataFrame()
  for i,j in zip(index_list,name_list):
      df = pro.index_daily(ts_code=i,start_date='20100101',fields=[
  
          "trade_date",
          "close"])
      df.loc[:,'trade_date']=pd.to_datetime(df.loc[:,'trade_date'])
      df=df.sort_values(by='trade_date')
      df=df.rename(columns={'close':j,'trade_date':'date'})
      df=df.set_index('date')
      all_df=pd.concat([all_df,df],axis=1)
  all_df1=pd.DataFrame()
  for x,y in zip(global_list,global_name):
      df = pro.index_global(ts_code=x,start_date='20100101',fields=[
  
          "trade_date",
          "close"])
      df.loc[:,'trade_date']=pd.to_datetime(df.loc[:,'trade_date'])
      df=df.sort_values(by='trade_date')
      df=df.rename(columns={'close':y,'trade_date':'date'})
      df=df.set_index('date')
      all_df1=pd.concat([all_df1,df],axis=1)
  global_index_df=pd.concat([all_df,all_df1],axis=1).fillna(method='pad',axis=0)
   
  bond_df = ak.bond_new_composite_index_cbond(indicator="财富", period="总值").rename(columns={'value':'中债财富总值'})
  bond_df['date']=pd.to_datetime(bond_df['date'])
  bond_df=bond_df.set_index('date')
  df_all_=pd.concat([global_index_df,bond_df],axis=1).fillna(method='pad',axis=0).dropna()

    return df_all_
    
df_all=get_data()

start_date = st.date_input(
    "请选择开始日期",
    date(2020,2,9))
start=str(start_date)
end_date = st.date_input(
    "请选择结束日期",
    date(2021,5,9))
end=str(end_date)

options = st.multiselect(
    '请选择资产名称（多选）',
    ['纳斯达克','标普500','恒生指数','日经225','富时100','德国DAX30','沪深300','中证500','上证50','中证1000','中证800','中债财富总值']
)

method = st.selectbox(
    '请选择资产配置方法',
    ('等权重','等波动','Global Minimum Variance','风险平价','风险平价-压缩估计量','风险平价-衰减加权','风险平价-衰减加权-压缩估计量','下行波动率','下行波动率-衰减加权'))
freq=st.selectbox(
    '请选择调仓频率',
    ('月频','半年度','年度'))

#等权重计算方法
def equal_weights(datas,period='month'):
    ret = datas.pct_change(1).fillna(0)
    data_norm = datas/datas.iloc[0,]*1000
    result = data_norm.copy()    
    result['m'] = result.index
    result['m'] = result.m.apply(lambda x:x.month)
    weights = pd.DataFrame(columns = datas.columns,index = datas.index).fillna(0)
    N = pd.DataFrame(columns = datas.columns,index = datas.index).fillna(0)
    if period == 'month':
        for i in range(result.shape[0]):
            if i == 0:
                weights.iloc[i,:] = 1/datas.shape[1]   
                price = datas.loc[datas.index[i],:]
                n = list(np.divide(np.array(weights.iloc[i,:].values),np.array(price.values)))                       
                N.loc[result.index[i],:] = n                
            elif result.m[i] != result.m[i - 1]:
                weights.iloc[i,:] = 1/datas.shape[1]                
                price = datas.loc[datas.index[i],:]
                n =list(np.divide(np.array(weights.iloc[i,:].values),np.array(price.values)))                          
                N.loc[result.index[i],:] = n                                 
            else:
                N.iloc[i,:] = N.iloc[i-1,:]
                w = np.multiply(np.array(N.iloc[i,:]),np.array(datas.loc[datas.index[i],:]))                 
                weights.iloc[i,:] = list(np.divide(w,np.sum(w)))      
                
    elif period == '6month':
        for i in range(result.shape[0]):
            if i == 0:
                weights.iloc[i,:] = 1/datas.shape[1]   
                price = datas.loc[datas.index[i],:]
                n = list(np.divide(np.array(weights.iloc[i,:].values),np.array(price.values)))      
                N.loc[result.index[i],:] = n                  
            elif (result.m[i] != result.m[i - 1] and  result.m[i]%6==0) :
                weights.iloc[i,:] = 1/datas.shape[1]                
                price = datas.loc[datas.index[i],:]
                n = list(np.divide(np.array(weights.iloc[i,:].values),np.array(price.values)))                         
                N.loc[result.index[i],:] = n  
            else:
                N.iloc[i,:] = N.iloc[i-1,:]
                w = np.multiply(np.array(N.iloc[i,:]),np.array(datas.loc[datas.index[i],:]))                
                weights.iloc[i,:] = list(np.divide(w,np.sum(w)))   
    elif period == 'year':
        for i in range(result.shape[0]):
            if i == 0 :
                weights.iloc[i,:] = 1/datas.shape[1]   
                price = datas.loc[datas.index[i],:]
                n =list(np.divide(np.array(weights.iloc[i,:].values),np.array(price.values)))       
                N.loc[result.index[i],:] = n           
            elif (result.m[i] != result.m[i - 1] and  result.m[i]%12==1) :
                weights.iloc[i,:] = 1/datas.shape[1]                
                price = datas.loc[datas.index[i],:]
                n = list(np.divide(np.array(weights.iloc[i,:].values),np.array(price.values)))                         
                N.loc[result.index[i],:] = n  
            else:
                N.iloc[i,:] = N.iloc[i-1,:]
                w = np.multiply(np.array(N.iloc[i,:]),np.array(datas.loc[datas.index[i],:]))           
                weights.iloc[i,:] = list(np.divide(w,np.sum(w)))        

    else: 
        return '请输入调仓周期'
    
    result['mv'] = 0
    result['mv_adj_last_day'] = 0
    result['nav'] = 1
    for i in range(result.shape[0]):
        result.loc[result.index[i],'mv'] =np.sum(np.multiply(np.array(datas.iloc[i,:]),np.array(N.iloc[i,:])))
        if i == 0:
            pass
        elif all(N.iloc[i,:] == N.iloc[i-1,:]):
            result.loc[result.index[i],'mv_adj_last_day'] = result.loc[result.index[i-1],'mv']
            result.loc[result.index[i],'nav'] = np.divide(np.multiply(np.array(result.nav[i-1]),np.array(result.mv[i])),np.array(result.mv_adj_last_day[i]))
        else:
            result.loc[result.index[i],'mv_adj_last_day'] = np.sum(np.multiply(np.array(datas.iloc[i-1,:]),np.array(N.iloc[i,:])))
            result.loc[result.index[i],'nav'] = np.divide(np.multiply(np.array(result.nav[i-1]),np.array(result.mv[i])),np.array(result.mv_adj_last_day[i]))
    result['nav'] = result.nav/result.nav[0]*1000    
    return weights,result

#等波动计算方法
def EqualVolWeight(datas,period ='month'):
    ret = datas.pct_change(1).fillna(0)
    data_norm = datas/datas.iloc[0,]*1000
    result = data_norm.copy()    
    result['m'] = result.index
    result['m'] = result.m.apply(lambda x:x.month)
    weights = pd.DataFrame(columns = datas.columns,index = datas.index).fillna(0)
    N = pd.DataFrame(columns = datas.columns,index = datas.index).fillna(0)    
    position = 0
    if period == 'month':
        for i in range(result.shape[0]):
            if i == 0:
                pass
            elif result.m[i] != result.m[i - 1]:
                vol = ret.iloc[position:i].std()
                position = i
                weights.iloc[i,:] = (1/vol)/((1/vol).sum()) 
                price = datas.loc[datas.index[i],:]
                V = np.sum(np.multiply(np.array(weights.iloc[i,:]),np.array(price)))
                n =list(np.divide(np.multiply(np.array(weights.iloc[i,:].values),V),np.array(price.values)))
                N.loc[result.index[i],:] = n  
            else:
                N.iloc[i,:] = N.iloc[i-1,:]
                w = np.multiply(np.array(N.iloc[i,:]),np.array(datas.loc[datas.index[i],:]))
                weights.iloc[i,:] = list(np.divide(w,np.sum(w)))

    elif period == '6month':
        for i in range(result.shape[0]):
            if i == 0:
                pass
            elif (result.m[i] != result.m[i - 1] and  result.m[i]%6==0) :
                vol = ret.iloc[position:i].std()
                position = i
                weights.iloc[i,:] = (1/vol)/((1/vol).sum())  
                price = datas.loc[datas.index[i],:]
                V = np.sum(np.multiply(np.array(weights.iloc[i,:]),np.array(price)))
                n = list(np.divide(np.multiply(np.array(weights.iloc[i,:].values),V),np.array(price.values)))     
                N.loc[result.index[i],:] = n                  
            else:
                N.iloc[i,:] = N.iloc[i-1,:]
                w = np.multiply(np.array(N.iloc[i,:]),np.array(datas.loc[datas.index[i],:]))
                weights.iloc[i,:] = list(np.divide(w,np.sum(w)))

    elif period == 'year':
        for i in range(result.shape[0]):
            if i == 0:
                pass
            elif (result.m[i] != result.m[i - 1] and  result.m[i]%12==1) :
                vol = ret.iloc[position:i].std()
                position = i
                weights.iloc[i,:] = (1/vol)/((1/vol).sum()) 
                price = datas.loc[datas.index[i],:]
                V = np.sum(np.multiply(np.array(weights.iloc[i,:]),np.array(price)))
                n = list(np.divide(np.multiply(np.array(weights.iloc[i,:].values),V),np.array(price.values)))      
                N.loc[result.index[i],:] = n                  
            else:
                N.iloc[i,:] = N.iloc[i-1,:]
                w =  np.multiply(np.array(N.iloc[i,:]),np.array(datas.loc[datas.index[i],:]))
                weights.iloc[i,:] = list(np.divide(w,np.sum(w)))

    else: 
        return '请输入调仓周期'
    
    
    result['mv'] = 0
    result['mv_adj_last_day'] = 0
    result['nav'] = 1
    for i in range(result.shape[0]):
        result.loc[result.index[i],'mv'] =np.sum(np.multiply(np.array(datas.iloc[i,:]),np.array(N.iloc[i,:])))
        if all(N.iloc[i,:]==0):
            pass
        elif all(N.iloc[i,:] == N.iloc[i-1,:] ):             
            result.loc[result.index[i],'mv_adj_last_day'] = result.loc[result.index[i-1],'mv']
            result.loc[result.index[i],'nav'] = np.divide(np.multiply(np.array(result.nav[i-1]),np.array(result.mv[i])),np.array(result.mv_adj_last_day[i]))
        else:
            result.loc[result.index[i],'mv_adj_last_day'] = np.sum(np.multiply(np.array(datas.iloc[i-1,:]),np.array(N.iloc[i,:])))
            result.loc[result.index[i],'nav'] = np.divide(np.multiply(np.array(result.nav[i-1]),np.array(result.mv[i])),np.array(result.mv_adj_last_day[i]))            
    result['nav'] = result.nav/result.nav[0]*1000
        
    return weights,result

#Global Minimum Variance计算方法
def funs(weight,sigma):
    weight = np.array([weight]).T
    result = np.dot(np.dot(weight.T,np.mat(sigma)),weight)[0,0]
    return(result)
def ConstraintGMOWeight(datas,period ='month'):
    ret = datas.pct_change(1).fillna(0)
    data_norm = datas/datas.iloc[0,]*1000
    result = data_norm.copy()    
    result['m'] = result.index
    result['m'] = result.m.apply(lambda x:x.month)
    weights = pd.DataFrame(columns = datas.columns,index = datas.index).fillna(0)
    N = pd.DataFrame(columns = datas.columns,index = datas.index).fillna(0)    
    position = 0
    # 约束
    cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})
    # 边界
    bnds = tuple((0, 1) for i in range(datas.shape[1]))   
    
    if period == 'month':
        for i in range(result.shape[0]):
            if i == 0:
                pass
            elif result.m[i] != result.m[i - 1]:
                sigma = ret.iloc[position:i].cov()
                position = i              
                weight = [0 for i in range(datas.shape[1])]
                res =  minimize(funs,weight, method='SLSQP',args = (sigma,),
                bounds=bnds,constraints=cons,tol=1e-8)
                weights.iloc[i,:] =  res.x
                price = datas.loc[datas.index[i],:]
                V = np.sum(np.multiply(np.array(weights.iloc[i,:]),np.array(price)))
                n = list(np.divide(np.multiply(np.array(weights.iloc[i,:].values),V),np.array(price.values)))      
                N.loc[result.index[i],:] = n  
            else:

                N.iloc[i,:] = N.iloc[i-1,:]
                w = np.multiply(np.array(N.iloc[i,:]),np.array(datas.loc[datas.index[i],:]))
                weights.iloc[i,:] =list(np.divide(w,np.sum(w))) 
    elif period == '6month':
        for i in range(result.shape[0]):
            if i == 0:
                pass
            elif (result.m[i] != result.m[i - 1] and  result.m[i]%6==0) :
                sigma = ret.iloc[position:i].cov()
                position = i             
                weight = [0 for i in range(datas.shape[1])]
                res =  minimize(funs,weight, method='SLSQP',args = (sigma,),
                bounds=bnds,constraints=cons,tol=1e-8)
                weights.iloc[i,:] =  res.x
                price = datas.loc[datas.index[i],:]
                V = np.sum(np.multiply(np.array(weights.iloc[i,:]),np.array(price)))
                n = list(np.divide(np.multiply(np.array(weights.iloc[i,:].values),V),np.array(price.values)))      
                N.loc[result.index[i],:] = n                  
            else:
                N.iloc[i,:] = N.iloc[i-1,:]
                w = np.multiply(np.array(N.iloc[i,:]),np.array(datas.loc[datas.index[i],:]))
                
                weights.iloc[i,:] = list(np.divide(w,np.sum(w))) 
    elif period == 'year':
        for i in range(result.shape[0]):
            if i == 0:
                pass
            elif (result.m[i] != result.m[i - 1] and  result.m[i]%12==1) :
                sigma = ret.iloc[position:i].cov()
                position = i          
                weight = [0 for i in range(datas.shape[1])]
                res =  minimize(funs,weight, method='SLSQP',args = (sigma,),
                bounds=bnds,constraints=cons,tol=1e-8)
                weights.iloc[i,:] =  res.x
                price = datas.loc[datas.index[i],:]
                V = np.sum(np.multiply(np.array(weights.iloc[i,:]),np.array(price)))
                n = list(np.divide(np.multiply(np.array(weights.iloc[i,:].values),V),np.array(price.values)))      
                N.loc[result.index[i],:] = n                  
                
            else:
                N.iloc[i,:] = N.iloc[i-1,:]
                w = np.multiply(np.array(N.iloc[i,:]),np.array(datas.loc[datas.index[i],:])) 
                weights.iloc[i,:] = list(np.divide(w,np.sum(w))) 
    else: 
        return '请输入调仓周期'
    result['mv'] = 0
    result['mv_adj_last_day'] = 0
    result['nav'] = 1
    for i in range(result.shape[0]):
        result.loc[result.index[i],'mv'] = np.sum(np.multiply(np.array(datas.iloc[i,:]),np.array(N.iloc[i,:])))
        if all(N.iloc[i,:]==0):
            pass
        elif all(N.iloc[i,:] == N.iloc[i-1,:] ):             
            result.loc[result.index[i],'mv_adj_last_day'] = result.loc[result.index[i-1],'mv']
            result.loc[result.index[i],'nav'] = np.divide(np.multiply(np.array(result.nav[i-1]),np.array(result.mv[i])),np.array(result.mv_adj_last_day[i]))
        else:
            result.loc[result.index[i],'mv_adj_last_day'] =  np.sum(np.multiply(np.array(datas.iloc[i-1,:]),np.array(N.iloc[i,:])))
            result.loc[result.index[i],'nav'] = np.divide(np.multiply(np.array(result.nav[i-1]),np.array(result.mv[i])),np.array(result.mv_adj_last_day[i]))    
    result['nav'] = result.nav/result.nav[0]*1000
    return weights,result

#风险平价计算方法
def funsRP(weight,sigma):
    weight = np.array([weight]).T
    X = np.multiply(weight,np.dot(sigma.values,weight))
    result = np.square(np.dot(X,np.ones([1,X.shape[0]])) - X.T).sum()
    return(result)
def RPWeight(datas,period ='month'):
    ret = datas.pct_change(1).fillna(0)
    data_norm = datas/datas.iloc[0,]*1000
    result = data_norm.copy()    
    result['m'] = result.index
    result['m'] = result.m.apply(lambda x:x.month)
    weights = pd.DataFrame(columns = datas.columns,index = datas.index).fillna(0)
    N = pd.DataFrame(columns = datas.columns,index = datas.index).fillna(0)

    # 约束
    cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})
    # 边界
    bnds = tuple((0, 1) for i in range(datas.shape[1]))    
    position = 0
    
    if period == 'month':
        for i in range(result.shape[0]):
            if i == 0:
                pass
            elif result.m[i] != result.m[i - 1]:
                sigma = ret.iloc[position:i].cov()
                position = i              
                weight = [0 for i in range(datas.shape[1])]
                res =  minimize(funsRP,weight, method='SLSQP',args = (sigma,),
                bounds=bnds,constraints=cons,tol=1e-20)
                weights.iloc[i,:] =  res.x
                price = datas.loc[datas.index[i],:]
                V = np.sum(np.multiply(np.array(weights.iloc[i,:]),np.array(price)))
                n = list(np.divide(np.multiply(np.array(weights.iloc[i,:].values),V),np.array(price.values)))         
                N.loc[result.index[i],:] = n   
            else:
                N.iloc[i,:] = N.iloc[i-1,:]
                w = np.multiply(np.array(N.iloc[i,:]),np.array(datas.loc[datas.index[i],:]))
                weights.iloc[i,:] =list(np.divide(w,np.sum(w)))

    elif period == '6month':
        for i in range(result.shape[0]):
            if i == 0:
                pass
            elif (result.m[i] != result.m[i - 1] and  result.m[i]%6==0) :
                sigma = ret.iloc[position:i].cov()
                position = i              
                weight = [0 for i in range(datas.shape[1])]
                res =  minimize(funsRP,weight, method='SLSQP',args = (sigma,),
                bounds=bnds,constraints=cons,tol=1e-20)

                weights.iloc[i,:] =  res.x
                price = datas.loc[datas.index[i],:]
                V = np.sum(np.multiply(np.array(weights.iloc[i,:]),np.array(price)))
                n = list(np.divide(np.multiply(np.array(weights.iloc[i,:].values),V),np.array(price.values)))       
                N.loc[result.index[i],:] = n                   
                
            else:
                N.iloc[i,:] = N.iloc[i-1,:]
                w = np.multiply(np.array(N.iloc[i,:]),np.array(datas.loc[datas.index[i],:]))
                weights.iloc[i,:] =list(np.divide(w,np.sum(w)))

    elif period == 'year':
        for i in range(result.shape[0]):
            if i == 0:
                pass
            elif (result.m[i] != result.m[i - 1] and  result.m[i]%12==1) :
                sigma = ret.iloc[position:i].cov()
                position = i           
                weight = [0 for i in range(datas.shape[1])]
                res =  minimize(funsRP,weight, method='SLSQP',args = (sigma,),
                bounds=bnds,constraints=cons,tol=1e-20)
                weights.iloc[i,:] =  res.x
                price = datas.loc[datas.index[i],:]
                V = np.sum(np.multiply(np.array(weights.iloc[i,:]),np.array(price)))
                n = list(np.divide(np.multiply(np.array(weights.iloc[i,:].values),V),np.array(price.values)))          
                N.loc[result.index[i],:] = n   
            else:
                N.iloc[i,:] = N.iloc[i-1,:]
                w = np.multiply(np.array(N.iloc[i,:]),np.array(datas.loc[datas.index[i],:]))
                weights.iloc[i,:] =list(np.divide(w,np.sum(w))) 

    else: 
        return '请输入调仓周期'
    
    result['mv'] = 0
    result['mv_adj_last_day'] = 0
    result['nav'] = 1
    for i in range(result.shape[0]):
        result.loc[result.index[i],'mv'] = np.sum(np.multiply(np.array(datas.iloc[i,:]),np.array(N.iloc[i,:])))
        if all(N.iloc[i,:]==0):
            pass
        elif all(N.iloc[i,:] == N.iloc[i-1,:] ):             
            result.loc[result.index[i],'mv_adj_last_day'] = result.loc[result.index[i-1],'mv']
            result.loc[result.index[i],'nav'] = np.divide(np.multiply(np.array(result.nav[i-1]),np.array(result.mv[i])),np.array(result.mv_adj_last_day[i]))
    
        else:
            
            result.loc[result.index[i],'mv_adj_last_day'] = np.sum(np.multiply(np.array(datas.iloc[i-1,:]),np.array(N.iloc[i,:])))
            result.loc[result.index[i],'nav'] = np.divide(np.multiply(np.array(result.nav[i-1]),np.array(result.mv[i])),np.array(result.mv_adj_last_day[i]))         
    result['nav'] = result.nav/result.nav[0]*1000

    return weights,result

#风险平价-压缩估计计算
def RPLedoit(datas,period ='month'):
    ret = datas.pct_change(1).fillna(0)
    data_norm = datas/datas.iloc[0,]*1000
    result = data_norm.copy()    
    result['m'] = result.index
    result['m'] = result.m.apply(lambda x:x.month)
    weights = pd.DataFrame(columns = datas.columns,index = datas.index).fillna(0)
    N = pd.DataFrame(columns = datas.columns,index = datas.index).fillna(0)

    # 约束
    cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})
    # 边界
    bnds = tuple((0, 1) for i in range(datas.shape[1]))    
    position = 0
    
    if period == 'month':
        for i in range(result.shape[0]):
            if i == 0:
                pass
            elif result.m[i] != result.m[i - 1]:
               # sigma = ret.iloc[position:i].cov()
                sigma,a = ledoit_wolf(ret.iloc[position:i])
                sigma = pd.DataFrame(sigma)
                position = i              
                weight = [0 for i in range(datas.shape[1])]
                res =  minimize(funsRP,weight, method='SLSQP',args = (sigma,),
                bounds=bnds,constraints=cons,tol=1e-20)
                weights.iloc[i,:] =  res.x
                price = datas.loc[datas.index[i],:]
                V = np.sum(np.multiply(np.array(weights.iloc[i,:]),np.array(price)))
                n = list(np.divide(np.multiply(np.array(weights.iloc[i,:].values),V),np.array(price.values)))       
                N.loc[result.index[i],:] = n   
            else:
                N.iloc[i,:] = N.iloc[i-1,:]
                w = np.multiply(np.array(N.iloc[i,:]),np.array(datas.loc[datas.index[i],:]))
                weights.iloc[i,:] =list(np.divide(w,np.sum(w)))

    elif period == '6month':
        for i in range(result.shape[0]):
            if i == 0:
                pass
            elif (result.m[i] != result.m[i - 1] and  result.m[i]%6==0) :
              #  sigma = ret.iloc[position:i].cov()
                sigma,a = ledoit_wolf(ret.iloc[position:i])
                sigma = pd.DataFrame(sigma)
                position = i              
                weight = [0 for i in range(datas.shape[1])]
                res =  minimize(funsRP,weight, method='SLSQP',args = (sigma,),
                bounds=bnds,constraints=cons,tol=1e-20)

                weights.iloc[i,:] =  res.x
                price = datas.loc[datas.index[i],:]
                V = np.sum(np.multiply(np.array(weights.iloc[i,:]),np.array(price)))
                n = list(np.divide(np.multiply(np.array(weights.iloc[i,:].values),V),np.array(price.values)))      
                N.loc[result.index[i],:] = n                   
                
            else:
                N.iloc[i,:] = N.iloc[i-1,:]
                w = np.multiply(np.array(N.iloc[i,:]),np.array(datas.loc[datas.index[i],:]))
                weights.iloc[i,:] =list(np.divide(w,np.sum(w)))  

    elif period == 'year':
        for i in range(result.shape[0]):
            if i == 0:
                pass
            elif (result.m[i] != result.m[i - 1] and  result.m[i]%12==1) :
                #sigma = ret.iloc[position:i].cov()
                sigma,a = ledoit_wolf(ret.iloc[position:i])
                sigma = pd.DataFrame(sigma)
                position = i         
                weight = [0 for i in range(datas.shape[1])]
                res =  minimize(funsRP,weight, method='SLSQP',args = (sigma,),
                bounds=bnds,constraints=cons,tol=1e-20)
                weights.iloc[i,:] =  res.x
                price = datas.loc[datas.index[i],:]
                V = np.sum(np.multiply(np.array(weights.iloc[i,:]),np.array(price)))
                n = list(np.divide(np.multiply(np.array(weights.iloc[i,:].values),V),np.array(price.values)))      
                N.loc[result.index[i],:] = n   
            else:
                N.iloc[i,:] = N.iloc[i-1,:]
                w = np.multiply(np.array(N.iloc[i,:]),np.array(datas.loc[datas.index[i],:]))
                weights.iloc[i,:] =list(np.divide(w,np.sum(w)))

    else: 
        return '请输入调仓周期'
    
    result['mv'] = 0
    result['mv_adj_last_day'] = 0
    result['nav'] = 1
    for i in range(result.shape[0]):
        result.loc[result.index[i],'mv'] = np.sum(np.multiply(np.array(datas.iloc[i,:]),np.array(N.iloc[i,:])))
        if all(N.iloc[i,:]==0):
            pass
        elif all(N.iloc[i,:] == N.iloc[i-1,:] ):             
            result.loc[result.index[i],'mv_adj_last_day'] = result.loc[result.index[i-1],'mv']
            result.loc[result.index[i],'nav'] = np.divide(np.multiply(np.array(result.nav[i-1]),np.array(result.mv[i])),np.array(result.mv_adj_last_day[i]))
        else:
            result.loc[result.index[i],'mv_adj_last_day'] = np.sum(np.multiply(np.array(datas.iloc[i-1,:]),np.array(N.iloc[i,:])))
            result.loc[result.index[i],'nav'] = np.divide(np.multiply(np.array(result.nav[i-1]),np.array(result.mv[i])),np.array(result.mv_adj_last_day[i]))          
    result['nav'] = result.nav/result.nav[0]*1000

    return weights,result

def getSigma(datas,method = 'Simple'):
    asset = datas.columns
    datas['n'] = np.arange(datas.shape[0])
    datas['group'] = pd.qcut(datas.n,4,labels = False)
    weights = np.arange(1,5)/10
    
    if method == 'Simple':
        sigma_1 = datas.loc[datas.group==0,asset].cov()
        sigma_2 = datas.loc[datas.group==1,asset].cov()
        sigma_3 = datas.loc[datas.group==2,asset].cov()
        sigma_4 = datas.loc[datas.group==3,asset].cov()
        sigma = 0.1*sigma_1 +sigma_2*0.2 +sigma_3*0.3 +sigma_4*0.4
    elif method =='Ledoit':
        sigma_1,a = ledoit_wolf(datas.loc[datas.group==0,asset])
        sigma_2,a = ledoit_wolf(datas.loc[datas.group==1,asset])
        sigma_3,a = ledoit_wolf(datas.loc[datas.group==2,asset])
        sigma_4,a = ledoit_wolf(datas.loc[datas.group==3,asset])
        sigma = 0.1*sigma_1 +sigma_2*0.2 +sigma_3*0.3 +sigma_4*0.4
        sigma = pd.DataFrame(sigma)
    elif method == 'DW':
        datas[datas>0] = 0
        datas['n'] = np.arange(datas.shape[0])
        datas['group'] = pd.qcut(datas.n,4,labels = False)        
        sigma_1 = datas.loc[datas.group==0,asset].cov()
        sigma_2 = datas.loc[datas.group==1,asset].cov()
        sigma_3 = datas.loc[datas.group==2,asset].cov()
        sigma_4 = datas.loc[datas.group==3,asset].cov()
        sigma = 0.1*sigma_1 +sigma_2*0.2 +sigma_3*0.3 +sigma_4*0.4        
    else:
        pass
    return sigma

#风险平价-半衰期加权
def RPHalfWeight(datas,period ='month',method = 'Simple'):
    ret = datas.pct_change(1).fillna(0)
    data_norm = datas/datas.iloc[0,]*1000
    result = data_norm.copy()    
    result['m'] = result.index
    result['m'] = result.m.apply(lambda x:x.month)
    weights = pd.DataFrame(columns = datas.columns,index = datas.index).fillna(0)
    N = pd.DataFrame(columns = datas.columns,index = datas.index).fillna(0)
    # 约束
    cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})
    # 边界
    bnds = tuple((0,1) for i in range(datas.shape[1]))    
    position = 0
    if period == 'month':
        for i in range(result.shape[0]):
            if i == 0:
                pass
            elif result.m[i] != result.m[i - 1]:
                sigma =  getSigma(ret.iloc[position:i],method = method)
                position = i              
                weight = [0 for i in range(datas.shape[1])]
                res =  minimize(funsRP,weight, method='SLSQP',args = (sigma,),
                bounds=bnds,constraints=cons,tol=1e-20)
                weights.iloc[i,:] =  res.x
                price = datas.loc[datas.index[i],:]
                V = np.sum(np.multiply(np.array(weights.iloc[i,:]),np.array(price)))
                n = list(np.divide(np.multiply(np.array(weights.iloc[i,:].values),V),np.array(price.values)))      
                N.loc[result.index[i],:] = n   
            else:
                N.iloc[i,:] = N.iloc[i-1,:]
                w = np.multiply(np.array(N.iloc[i,:]),np.array(datas.loc[datas.index[i],:]))
                weights.iloc[i,:] =list(np.divide(w,np.sum(w))) 

    elif period == '6month':
        for i in range(result.shape[0]):
            if i == 0:
                pass
            elif (result.m[i] != result.m[i - 1] and  result.m[i]%6==0) :
                sigma  =  getSigma(ret.iloc[position:i],method = method)
                position = i              
                weight = [0 for i in range(datas.shape[1])]
                res =  minimize(funsRP,weight, method='SLSQP',args = (sigma,),
                bounds=bnds,constraints=cons,tol=1e-20)

                weights.iloc[i,:] =  res.x
                price = datas.loc[datas.index[i],:]
                V = np.sum(np.multiply(np.array(weights.iloc[i,:]),np.array(price)))
                n = list(np.divide(np.multiply(np.array(weights.iloc[i,:].values),V),np.array(price.values)))   
                N.loc[result.index[i],:] = n                   
                
            else:
                N.iloc[i,:] = N.iloc[i-1,:]
                w = np.multiply(np.array(N.iloc[i,:]),np.array(datas.loc[datas.index[i],:]))
                weights.iloc[i,:] =list(np.divide(w,np.sum(w)))  

    elif period == 'year':
        for i in range(result.shape[0]):
            if i == 0:
                pass
            elif (result.m[i] != result.m[i - 1] and  result.m[i]%12==1) :
                sigma  =  getSigma(ret.iloc[position:i],method = method)
                position = i         
                weight = [0 for i in range(datas.shape[1])]
                res =  minimize(funsRP,weight, method='SLSQP',args = (sigma,),
                bounds=bnds,constraints=cons,tol=1e-20)
                weights.iloc[i,:] =  res.x
                price = datas.loc[datas.index[i],:]
                V = np.sum(np.multiply(np.array(weights.iloc[i,:]),np.array(price)))
                n = list(np.divide(np.multiply(np.array(weights.iloc[i,:].values),V),np.array(price.values)))   
                N.loc[result.index[i],:] = n   
            else:
                N.iloc[i,:] = N.iloc[i-1,:]
                w = np.multiply(np.array(N.iloc[i,:]),np.array(datas.loc[datas.index[i],:]))
                weights.iloc[i,:] =list(np.divide(w,np.sum(w)))

    else: 
        return '请输入调仓周期'
    
    result['mv'] = 0
    result['mv_adj_last_day'] = 0
    result['nav'] = 1
    for i in range(result.shape[0]):
        result.loc[result.index[i],'mv'] = np.sum(np.multiply(np.array(datas.iloc[i,:]),np.array(N.iloc[i,:])))
        if all(N.iloc[i,:]==0):
            pass
        elif all(N.iloc[i,:] == N.iloc[i-1,:] ):             
            result.loc[result.index[i],'mv_adj_last_day'] = result.loc[result.index[i-1],'mv']
            result.loc[result.index[i],'nav'] = np.divide(np.multiply(np.array(result.nav[i-1]),np.array(result.mv[i])),np.array(result.mv_adj_last_day[i]))
        else:
            result.loc[result.index[i],'mv_adj_last_day'] = np.sum(np.multiply(np.array(datas.iloc[i-1,:]),np.array(N.iloc[i,:])))
            result.loc[result.index[i],'nav'] = np.divide(np.multiply(np.array(result.nav[i-1]),np.array(result.mv[i])),np.array(result.mv_adj_last_day[i]))             
    result['nav'] = result.nav/result.nav[0]*1000

    return weights,result

#下行波动率计算
def RP_DownWard(datas,period ='month'):
    ret = datas.pct_change(1).fillna(0)
    data_norm = datas/datas.iloc[0,]*1000
    result = data_norm.copy()    
    result['m'] = result.index
    result['m'] = result.m.apply(lambda x:x.month)
    weights = pd.DataFrame(columns = datas.columns,index = datas.index).fillna(0)
    N = pd.DataFrame(columns = datas.columns,index = datas.index).fillna(0)

    # 约束
    cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})
    # 边界
    bnds = tuple((0,1) for i in range(datas.shape[1]))    
    position = 0
    if period == 'month':
        for i in range(result.shape[0]):
            if i == 0:
                pass
            elif result.m[i] != result.m[i - 1]:
                data_cov = ret.iloc[position:i]
                data_cov[data_cov>0] = 0 
                sigma = data_cov.cov()
                position = i              
                weight = [0 for i in range(datas.shape[1])]
                res =  minimize(funsRP,weight, method='SLSQP',args = (sigma,),
                bounds=bnds,constraints=cons,tol=1e-20)
                weights.iloc[i,:] =  res.x
                price = datas.loc[datas.index[i],:]
                V = np.sum(np.multiply(np.array(weights.iloc[i,:]),np.array(price)))
                n = list(np.divide(np.multiply(np.array(weights.iloc[i,:].values),V),np.array(price.values)))        
                N.loc[result.index[i],:] = n   
            else:
                N.iloc[i,:] = N.iloc[i-1,:]
                w = np.multiply(np.array(N.iloc[i,:]),np.array(datas.loc[datas.index[i],:]))
                weights.iloc[i,:] =list(np.divide(w,np.sum(w))) 

    elif period == '6month':
        for i in range(result.shape[0]):
            if i == 0:
                pass
            elif (result.m[i] != result.m[i - 1] and  result.m[i]%6==0) :
                data_cov = ret.iloc[position:i]
                data_cov[data_cov>0] = 0 
                sigma = data_cov.cov()
                position = i              
                weight = [0 for i in range(datas.shape[1])]
                res =  minimize(funsRP,weight, method='SLSQP',args = (sigma,),
                bounds=bnds,constraints=cons,tol=1e-20)

                weights.iloc[i,:] =  res.x
                price = datas.loc[datas.index[i],:]
                V = np.sum(np.multiply(np.array(weights.iloc[i,:]),np.array(price)))
                n = list(np.divide(np.multiply(np.array(weights.iloc[i,:].values),V),np.array(price.values)))      
                N.loc[result.index[i],:] = n                   
                
            else:
                N.iloc[i,:] = N.iloc[i-1,:]
                w = np.multiply(np.array(N.iloc[i,:]),np.array(datas.loc[datas.index[i],:]))
                weights.iloc[i,:] =list(np.divide(w,np.sum(w)))   

    elif period == 'year':
        for i in range(result.shape[0]):
            if i == 0:
                pass
            elif (result.m[i] != result.m[i - 1] and  result.m[i]%12==1) :
                data_cov = ret.iloc[position:i]
                data_cov[data_cov>0] = 0 
                sigma = data_cov.cov()
                position = i           
                weight = [0 for i in range(datas.shape[1])]
                res =  minimize(funsRP,weight, method='SLSQP',args = (sigma,),
                bounds=bnds,constraints=cons,tol=1e-20)
                weights.iloc[i,:] =  res.x
                price = datas.loc[datas.index[i],:]
                V = np.sum(np.multiply(np.array(weights.iloc[i,:]),np.array(price)))
                n = list(np.divide(np.multiply(np.array(weights.iloc[i,:].values),V),np.array(price.values)))      
                N.loc[result.index[i],:] = n   
            else:
                N.iloc[i,:] = N.iloc[i-1,:]
                w = np.multiply(np.array(N.iloc[i,:]),np.array(datas.loc[datas.index[i],:]))
                weights.iloc[i,:] =list(np.divide(w,np.sum(w)))  

    else: 
        return '请输入调仓周期'
    
    result['mv'] = 0
    result['mv_adj_last_day'] = 0
    result['nav'] = 1
    for i in range(result.shape[0]):
        result.loc[result.index[i],'mv'] = np.sum(np.multiply(np.array(datas.iloc[i,:]),np.array(N.iloc[i,:])))
        if all(N.iloc[i,:]==0):
            pass
        elif all(N.iloc[i,:] == N.iloc[i-1,:] ):             
            result.loc[result.index[i],'mv_adj_last_day'] = result.loc[result.index[i-1],'mv']
            result.loc[result.index[i],'nav'] = np.divide(np.multiply(np.array(result.nav[i-1]),np.array(result.mv[i])),np.array(result.mv_adj_last_day[i]))
        else:
            result.loc[result.index[i],'mv_adj_last_day'] =  np.sum(np.multiply(np.array(datas.iloc[i-1,:]),np.array(N.iloc[i,:])))
            result.loc[result.index[i],'nav'] = np.divide(np.multiply(np.array(result.nav[i-1]),np.array(result.mv[i])),np.array(result.mv_adj_last_day[i]))             
    result['nav'] = result.nav/result.nav[0]*1000

    return weights,result

if (freq)and(st.button('开始')):
    #('月频','半年度','年度')
    if freq=='月频':
        freq='month'
    elif freq=='半年度':
        freq='6month'
    else:
        freq='year'

    df_all1=df_all[start:end]
    df_all1=df_all1[options]
    #('等权重','等波动','Global Minimum Variance','风险平价','风险平价-压缩估计量','风险平价-衰减加权','风险平价-衰减加权-压缩估计量','下行波动率','下行波动率-衰减加权')
    if method=='等权重':
        weight,nav=equal_weights(df_all1,period =freq)
    elif method=='等波动':
        weight,nav=EqualVolWeight(df_all1,period =freq)
    elif method=='Global Minimum Variance':
        weight,nav=ConstraintGMOWeight(df_all1,period =freq)
    elif method=='风险平价':
        weight,nav=RPWeight(df_all1,period =freq)
    elif method=='风险平价-压缩估计量':
        weight,nav=RPLedoit(df_all1,period =freq)
    elif method=='风险平价-衰减加权':
        weight,nav=RPHalfWeight(df_all1,period =freq,method = 'Simple')
    elif method=='风险平价-衰减加权-压缩估计量':
        weight,nav=RPHalfWeight(df_all1,period =freq,method ='Ledoit')
    elif method=='下行波动率':
        weight,nav=RP_DownWard(df_all1,period=freq)
    else:
        weight,nav=RPHalfWeight(df_all1,period =freq,method ='DW')

    
    if len(weight)>0:
        weight=weight.fillna(0)
        for i in weight.columns:
            weight[i]=weight[i].apply(lambda x: round(x,3)).apply(lambda x: x*100)
            
        df_all2=df_all[start:end]
        nav1=nav['nav'].apply(lambda x: x/1000)
    
        df_all2['中证500']=df_all2['中证500']/df_all2['中证500'][0]
        nav2=df_all2['中证500']
        #绘制面积图
        x1 = list(nav1.index.strftime('%Y/%m/%d'))
        data_sw = [go.Scatter(name=i, x=x1, y=list(weight[i]), stackgroup="one") for i in list(weight.columns)]
        layout1 = go.Layout(
            title = '各资产配置比例图',
            showlegend = True,
            xaxis = dict(
                type = 'category',
            ),
            yaxis = dict(
                type = 'linear',
                range = [0, 100],
                dtick = 20
                
            )
        )

        fig2 = go.Figure(data = data_sw, layout = layout1)
        
        #绘制净值走势图
        
        nav_yiled_trace_v2 = go.Scatter(x=nav1.index.strftime('%Y/%m/%d'),
                                        y=list(nav2), mode='lines', name='中证500')
        CIStyfit_yield_trace_v2 = go.Scatter(x=nav1.index.strftime('%Y/%m/%d'),
                                            y=list(nav1), mode='lines', name=f'配置策略收益-调仓频率:{freq}',line = dict(color = ('rgb(205, 12, 24)')))
        fig_nav = go.Figure(
            data=[nav_yiled_trace_v2, CIStyfit_yield_trace_v2])

        fig_nav.update_layout(
            title_text="基准与配置策略净值比较图 <br> 最新净值日期:" +
            nav1.index[-1].strftime('%Y-%m-%d'),
            margin=dict(l=100, r=100, t=60, b=80),
            yaxis={'tickformat': '.2f', 'title': ' 净值'},
            xaxis={'tickangle': -70, 'tickformat': '%Y-%m-%d'},legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5))
        
        base_drown=-((nav2.cummax() - nav2) /
                                        (nav2.cummax()))
        base_drown=base_drown.apply(lambda x: round(x,2))
        stra=-((nav1.cummax() - nav1) /
                                        (nav1.cummax()))
        stra=stra.apply(lambda x: round(x,2))
        drow_down1 = go.Scatter(x=nav1.index.strftime('%Y/%m/%d'),
                                        y=list(base_drown), mode='lines', name='中证500回撤情况',fill='tozeroy')
        drow_down2 = go.Scatter(x=nav1.index.strftime('%Y/%m/%d'),
                                            y=list(stra), mode='lines', name=f'配置策略收益-调仓频率:{freq}回撤情况',fill='tozeroy',line = dict(color = ('rgb(205, 12, 24)')))
        fig_nav1 = go.Figure(
            data=[drow_down1, drow_down2])

        fig_nav1.update_layout(
            title_text="基准与配置策略回撤比较图 <br> 最新净值日期:" +
            nav1.index[-1].strftime('%Y-%m-%d'),legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5)
            )
        
        def performance(datas):
            nav = datas/datas[0]
            nav_pct=nav.pct_change()
            # 样本期的年化收益率
            annual_ret = pow(nav[-1]/nav[0], 250/(len(nav)-1))-1
            # 样本期的最大回撤
            interval_max_down =((nav.cummax()-nav) /
                                (nav.cummax())).max()
            # 样本期年化波动率
            annual_var = (nav.pct_change()).std(
                    ddof=1)*pow(250, 0.5)
            # 样本期间年化夏普，年化后的平均收益率-无风险利率 /年化后的波动率
            rf_rate=0.02
            annual_sharpe = round((pow((1+(nav.pct_change()).mean()), 250)-1-rf_rate)/annual_var,2)
            # 样本期卡玛比率
            interval_calmar = round(annual_ret/interval_max_down,2)
            #样本期间胜率计算
            victory_rate=len(nav_pct[nav_pct>0])/len(nav_pct)

            return annual_ret,interval_max_down,annual_var,annual_sharpe,interval_calmar,victory_rate
        base=performance(nav2)
        pt_st=performance(nav1)
        df=pd.DataFrame()
        df['名称']=['中证500','策略']
        df['年化收益率']=[base[0],pt_st[0]]
        df['最大回撤']=[base[1],pt_st[1]]
        df['年化波动率']=[base[2],pt_st[2]]
        df['年化夏普']=[base[3],pt_st[3]]
        df['卡玛比率']=[base[4],pt_st[4]]
        df['胜率']=[base[5],pt_st[5]]
        df['年化收益率']=df['年化收益率'].map(lambda x:format(x,'.2%'))
        df['最大回撤']=df['最大回撤'].map(lambda x:format(x,'.2%'))
        df['年化波动率']=df['年化波动率'].map(lambda x:format(x,'.2%'))
        df['胜率']=df['胜率'].map(lambda x:format(x,'.2%'))
        table=create_table(df)

        st.plotly_chart(fig2)
        st.plotly_chart(fig_nav)
        st.plotly_chart(fig_nav1)
        st.subheader('基准与策略指标分析对比表格')
        st.plotly_chart(table)


