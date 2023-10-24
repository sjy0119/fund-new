import streamlit as st
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime,date
from scipy.optimize import minimize
import akshare as ak
import plotly as py
import plotly.graph_objs as go

@st.cache_data(ttl=600)
def load_data():
    df=pd.read_csv("风格指数数据",index_col=0)
    pf=pd.read_csv("申万一级数据")
    df.index=pd.DatetimeIndex(df.index)
    return df,pf
all=load_data()
style_index=all[0]
sw_index=all[1]

index_name=list(sw_index['指数名称'].unique())
sw1=sw_index.loc[sw_index['指数名称']==index_name[0]][['日期','收盘价']].rename(columns={'收盘价':f"{index_name[0]}",'日期':'date'}).set_index('date')
for i in range(1,31):
    ph_=sw_index.loc[sw_index['指数名称']==index_name[i]][['日期','收盘价']].rename(columns={'收盘价':f"{index_name[i]}",'日期':'date'}).set_index('date')
    sw1=pd.merge(sw1,ph_,left_index=True,right_index=True,how='left')
for i in sw1.columns:
    sw1[i+'pct']=sw1[i].pct_change()
sw1_pct=sw1.iloc[:,-31:]
today=date.today()

year_month=[str(date.today()-relativedelta(months=i))[:-3] for i in range(1,8)]

def size_analy(df,start_date,end_date):

    ff=df.copy()
    ff=ff[start_date:end_date]
    b1 = np.array(ff['基金日收益率'])
    A1= np.array(ff.iloc[:,2:-1])
    num_x = np.shape(A1)[1]
    def my_func(x):
        ls = np.abs(b1-np.dot(A1,x))**2
        result = np.sum(ls)
        return result
    def g1(x):
        return np.sum(x) #sum of X >= 0
    def g2(x):
        return 1-np.sum(x) #sum of X <= 1
    cons = ({'type': 'ineq', 'fun': g1}
            ,{'type': 'eq', 'fun': g2})
    x0 = np.zeros(num_x)
    bnds = [(0,1)]
    for i in range(num_x-1):
        bnds.append((0,1))
    res = minimize(my_func, 
                bounds = bnds, x0=x0,
                constraints=cons)
    ph=dict()
    ph['大盘成长']=res.x[0]
    ph['大盘价值']=res.x[1]
    ph['中盘成长']=res.x[2]
    ph['中盘价值']=res.x[3]
    ph['小盘成长']=res.x[4]
    ph['小盘价值']=res.x[5]
    #ph['中债财富总值']=res.x[6]
    ph1=pd.DataFrame([ph])

    return ph1
 
def size_sw(df,start_date,end_date):

    lf=df.copy()
    lf=lf[start_date:end_date]
    b1 = np.array(lf['基金日收益率'])  # 因变量
    A1 = np.array(lf.iloc[:,:31])
    num_x = np.shape(A1)[1]
    def my_func(x):
        ls = np.abs(b1-np.dot(A1,x))**2
        result = np.sum(ls)
        return result
    def g1(x):
        return np.sum(x) #sum of X >= 0
    def g2(x):
        return 1-np.sum(x) #sum of X <= 1
    cons = ({'type': 'ineq', 'fun': g1}
            ,{'type': 'eq', 'fun': g2})
    x0 = np.zeros(num_x)
    bnds = [(0,1)]
    for i in range(num_x-1):
        bnds.append((0,1))
    res = minimize(my_func, 
                bounds = bnds, x0=x0,
                constraints=cons)
    
    ph=dict()
    for i,j in zip(index_name,range(31)):
            ph[i]=res.x[j]
    #ph['中债财富总值']=res.x[6]
    ph1=pd.DataFrame([ph])

    return ph1

def cal_sds(df):
    year_name=str(df.index.year.unique()[0])
    time_list=[('0'+str(i))[-2:] for i in list(df.index.month.unique())]
    def cal_index(df):
        ff=df.copy()
        b1 = np.array(ff['基金日收益率'])
        A1= np.array(ff.iloc[:,2:-1])
        num_x = np.shape(A1)[1]
        def my_func(x):
            ls = np.abs((b1-np.dot(A1,x))**2)
            result = np.sum(ls)
            return result
        def g1(x):
            return np.sum(x) #sum of X >= 0
        def g2(x):
            return 1-np.sum(x) #sum of X <= 1
        cons = ({'type': 'ineq', 'fun': g1}
                ,{'type': 'eq', 'fun': g2})
        x0 = np.zeros(num_x)
        bnds = [(0,1)]
        for i in range(num_x-1):
            bnds.append((0,1))
        res = minimize(my_func, 
                    bounds = bnds, x0=x0,
                    constraints=cons)
        ph=dict()
        ph['大盘成长']=res.x[0]
        ph['大盘价值']=res.x[1]
        ph['中盘成长']=res.x[2]
        ph['中盘价值']=res.x[3]
        ph['小盘成长']=res.x[4]
        ph['小盘价值']=res.x[5]
        #ph['中债财富总值']=res.x[6]
        ph1=pd.DataFrame([ph])

        ph1=ph1.apply(lambda x: x*100)

        return ph1
    al=[cal_index(df[year_name+'-'+i:year_name+'-'+i]) for i in time_list]
    ax=pd.concat(al)
    num=np.sqrt(np.sum([np.var(ax[i]) for i in ['大盘成长','大盘价值','中盘成长','中盘价值','小盘成长','小盘价值']]))

    return round(num,3)



st.title('基金风格及行业变化预测追踪 :blue[!] :sunglasses:')
code=st.text_input('请输入基金代码例如000001')
text1='**基于基金收益率威廉·夏普风格分析，实现对基金风格的高频跟踪**'
st.caption(text1)
text2='晨星风格箱法往往会考虑利用重仓股对基金风格进行分析，但股票基金每季度更新上一季度重仓股情况，频率较低且具有一定的滞后性.\
       同时当重仓股权重不高时，估计的基金风格代表性不足，此时可以考虑利用基金收益率对基金风格进行判断，以增强结果的时效性与代表性。\
       威廉·夏普于 1992 年提出基于收益率的风格分析法，主要思想是设立一系列风格指数，利用最优化的方法，\
       最小化基金收益率与系列风格指数收益率的残差平方和，得到股票基金相对于各风格指数的暴露.'
st.caption(text2)


if st.button('开始运行'):
    
    fund_df = ak.fund_open_fund_info_em(fund=code, indicator="累计净值走势")
    fund_df['基金日收益率']=fund_df['累计净值'].pct_change().fillna(0)
    fund_df['净值日期']=pd.to_datetime(fund_df['净值日期'])
    fund_df=fund_df.rename(columns={'净值日期':'date'}).set_index('date')
    
    f_new=pd.merge(fund_df,style_index,left_index=True,right_index=True,how='left').dropna()

    year_list=list(f_new.index.year.unique())
    a_df=[f_new[str(i):str(i)] for i in year_list]

    ax=[ cal_sds(i) for i in a_df]

    
    df=[size_analy(f_new,year_month[i+1],year_month[i]) for i in range(6)]
    #df.append(fg)

    re_df=pd.concat(df,ignore_index=True)
    re_df=re_df.apply(lambda x: x*100)
    re_df['日期']=year_month[:-1]

    fi=re_df[['日期','大盘成长','大盘价值','中盘成长','中盘价值','小盘成长','小盘价值']]
    #fi=fi[['大盘成长','大盘价值','中盘成长','中盘价值','小盘成长','小盘价值']].apply(lambda x: x*100)

    sw1_pct.index=pd.DatetimeIndex(sw1_pct.index)
    f_sw=pd.merge(sw1_pct,fund_df,left_index=True,right_index=True,how='left').dropna()

    
    df1 = [size_sw(f_sw,year_month[i+1],year_month[i]) for i in range(6)]
        
    #df1=Parallel(n_jobs=6)(delayed(size_sw)(f_sw,year_month[i+1],year_month[i]) for i in range(6))
    #df1=[size_sw(f_sw,year_month[i+1],year_month[i]) for i in range(6)]
    sw_df=pd.concat(df1,ignore_index=True)
    sw_df=sw_df.apply(lambda x: x*100)
    sw_df['日期']=year_month[:-1]

    fi_new=sw_df[['日期']+index_name]


    x = list(fi['日期'])
    data_1= go.Scatter(name="大盘成长", x=x, y=list(fi['大盘成长']), stackgroup="one")
    data_2= go.Scatter(name="大盘价值", x=x, y=list(fi['大盘价值']), stackgroup="one")
    data_3= go.Scatter(name="中盘成长", x=x, y=list(fi['中盘成长']), stackgroup="one")
    data_4= go.Scatter(name="中盘价值", x=x, y=list(fi['中盘价值']), stackgroup="one")
    data_5= go.Scatter(name="小盘成长", x=x, y=list(fi['小盘成长']), stackgroup="one")
    data_6= go.Scatter(name="小盘价值", x=x, y=list(fi['小盘价值']), stackgroup="one")

    data = [data_1, data_2, data_3, data_4,data_5,data_6]
    layout = go.Layout(
        title = '市值风格变化比例图',
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

    fig = go.Figure(data = data, layout = layout)

    x1 = list(fi_new['日期'])
    data_sw = [go.Scatter(name=i, x=x1, y=list(fi_new[i]), stackgroup="one") for i in index_name]
    layout1 = go.Layout(
        title = '行业预测变化比例图',
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

    fig3= go.Figure(data = (go.Scatter(x=year_list,  
                                    y=ax,mode='lines')))
    fig3.update_layout(title_text='风格稳定情况')


    

    st.dataframe(fi)

    st.plotly_chart(fig)

    st.plotly_chart(fig2)

    st.caption('数值越大表明风格漂移越严重')
    st.plotly_chart(fig3)
   
    st.dataframe(fi_new)



