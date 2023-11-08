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
import requests
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] =False 


#开始和结束日期的选择
start_date = st.date_input(
    "请选择开始日期",
    date(2020,9,15))
start=str(start_date)
end_date = st.date_input(
    "请选择结束日期",
    date(2023,10,12))
end=str(end_date)

#全市场基金代码及简称数据
@st.cache_data
def load_data():
    df=pd.read_csv(r"C:\Users\WuKangmin\Desktop\全市场基金",index_col=0)
    df['基金代码']=df['基金代码'].apply(lambda x:('0000'+str(x))[-6:])
    return df
fund_=load_data()

#获取基准数据，以中证500为基准
@st.cache_data
def load_base():
    url='http://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.000905&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&lmt=58&klt=101&fqt=1&beg=20070115&end=20500101&ut=4f1862fc3b5e77c150a2b985b12db0fd&cb=cb_1699079114473_83662397&cb_1699079114473_83662397=cb_1699079114473_83662397'
    r=requests.get(url)
    data_text = r.text
    df=eval(data_text[26:-2])
    temp_df=pd.DataFrame([item.split(",") for item in df["data"]["klines"]]).iloc[:,:5]
    temp_df.columns = ["date", "open", "close", "high", "low"]
    temp_df=temp_df[['date','close']]
    temp_df['date']=pd.to_datetime(temp_df['date'])
    temp_df["close"] = pd.to_numeric(temp_df["close"], errors="coerce")
    temp_df=temp_df.rename(columns={'close':'中证500'})
    temp_df=temp_df.set_index('date')
    return temp_df
base_data=load_base()

#基金多选
options = st.multiselect(
    '请选择基金名称（多选）',
    [i for i in list(fund_['基金简称'])]
)
#资产配置方法选择
method = st.selectbox(
    '请选择资产配置方法',
    ('等权重','等波动','Global Minimum Variance','风险平价','风险平价-压缩估计量','风险平价-衰减加权','风险平价-衰减加权-压缩估计量','下行波动率','下行波动率-衰减加权'))
#调仓频率选择
freq=st.selectbox(
    '请输入调仓频率',('月频','半年度','年度'))

#等权重计算方法
def equal_weights(datas,period='month'):
    ret = datas.pct_change(1).fillna(0)
    data_norm = datas/datas.iloc[0,:]*1000
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
    N=N.fillna(0)
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
    data_norm = datas/datas.iloc[0,:]*1000
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
    
    N=N.fillna(0)
    result['mv'] = 0
    result['mv_adj_last_day'] = 0
    result['nav'] = 1
    for i in range(result.shape[0]):
        result.loc[result.index[i],'mv'] =np.sum(np.multiply(np.array(datas.iloc[i,:]),np.array(N.iloc[i,:])))
        result['mv']=result['mv'].fillna(0)
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
    data_norm = datas/datas.iloc[0,:]*1000
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
    N=N.fillna(0)
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
    data_norm = datas/datas.iloc[0,:]*1000
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
    N=N.fillna(0)
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
    data_norm = datas/datas.iloc[0,:]*1000
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
    N=N.fillna(0)
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

#设置不同协方差的计算方法
def getSigma(datas,method = 'Simple'):
    asset = datas.columns
    datas['n'] = np.arange(datas.shape[0])
    datas['group'] = pd.qcut(datas.n,4,labels = False,duplicates='drop')
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
        datas['group'] = pd.qcut(datas.n,4,labels = False,duplicates='drop')        
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
    data_norm = datas/datas.iloc[0,:]*1000
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
    N=N.fillna(0)
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
    data_norm = datas/datas.iloc[0,:]*1000
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
    N=N.fillna(0)
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

#异步爬虫爬取数据函数
def get(fund_id):
    dat=[]
    async def async_get_url(fund):
        
        url = f"http://fund.eastmoney.com/pingzhongdata/{fund}.js"  # 各类数据都在里面
        headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36"
            }
        async with aiohttp.ClientSession() as session:
            # 解释1
            async with session.get(url,headers=headers) as r:
                
                data_text = await r.text()
                
                try:
                    data_json = demjson.decode(
                        data_text[
                            data_text.find("Data_ACWorthTrend")
                            + 20 : data_text.find("Data_grandTotal")
                            - 16
                        ]
                    )
                except:
                    return pd.DataFrame()
                temp_df = pd.DataFrame(data_json)
                if temp_df.empty:
                    return pd.DataFrame()
                temp_df.columns = ["x", "y"]
                temp_df["x"] = pd.to_datetime(
                    temp_df["x"], unit="ms", utc=True
                ).dt.tz_convert("Asia/Shanghai")
                temp_df["x"] = temp_df["x"].dt.date
                temp_df.columns = [
                    "净值日期",
                    "累计净值",
                ]
                temp_df = temp_df[
                    [
                        "净值日期",
                        "累计净值",
                    ]
                ]
                temp_df["净值日期"] = pd.to_datetime(temp_df["净值日期"]).dt.date
                temp_df["累计净值"] = pd.to_numeric(temp_df["累计净值"])
                temp_df=temp_df.rename(columns={'累计净值':fund})
                temp_df=temp_df.set_index('净值日期')
                dat.append(temp_df)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = [async_get_url(fund) for fund in fund_id]
    loop.run_until_complete(asyncio.wait(tasks))
    return dat
#运行函数
@st.cache_data(ttl=600)
def get_data(fund_id):
    df=get(fund_id)
    df_all=pd.concat(df,axis=1).reset_index('净值日期').sort_values(by='净值日期').set_index('净值日期').dropna()
    df_all.index=pd.DatetimeIndex(df_all.index)
    return df_all

#运行部分
if (freq)and(st.button('开始')):

    fund_id=list(fund_.loc[fund_['基金简称'].isin(options)]['基金代码'])
    df_all=get_data(fund_id)

    st.caption('组合开始时间:'+df_all.index[0].strftime('%Y-%m-%d'))
    st.caption('组合结束时间:'+df_all.index[-1].strftime('%Y-%m-%d'))

    df_all1=df_all[start:end]
    if len(df_all1)==0:
        st.warning('请正确选择回测时间范围，确保回测数据有足够的时间长度')
    else:
        if freq=='月频':
            freq='month'
        elif freq=='半年度':
            freq='6month'
        else:
            freq='year'

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
            for i in weight.columns:
                weight[i]=weight[i].apply(lambda x: round(x,3)).apply(lambda x: x*100)
                
            base_data=base_data[start:end]
            nav1=nav['nav'].apply(lambda x: x/1000)

            base_data['中证500']=base_data['中证500']/base_data['中证500'][0]
            nav2=base_data['中证500']
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
                xaxis={'tickangle': -70, 'tickformat': '%Y-%m-%d'})
            
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
                nav1.index[-1].strftime('%Y-%m-%d')
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





