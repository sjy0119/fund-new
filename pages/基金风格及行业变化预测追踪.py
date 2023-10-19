import streamlit as st
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime,date
from scipy.optimize import minimize
import akshare as ak
from joblib import Parallel, delayed
import plotly as py
import plotly.graph_objs as go

@st.cache_data
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
    Y = ff['基金日收益率']  # 因变量
    # 使用最小二乘逼近求解，定义目标函数，minimize误差平方和
    # 多元线性模型中含常数项，待估参数分别为beta0，beta1，beta2，beta3....

    def fun(beta, ff): return beta[0] * ff['大盘成长pct'] + \
        beta[1] * ff['大盘价值pct'] + beta[2] * ff['中盘成长pct']+beta[3]*ff['中盘价值pct']+ beta[4] * ff['小盘成长pct']+beta[5]*ff['小盘价值pct']
    def objfunc(beta, Y, ff): return np.sum(np.abs(Y - fun(beta, ff)) ** 2)#+lam*sum([pow(beta[n], 2) for n in range(1,4)])
    # 输入变量的边界条件,自变量的回归系数取值在（0，1）

    bnds = ((0, 1), (0, 1), (0, 1),(0, 1), (0, 1), (0, 1))
    # 设置约束条件，回归系数之和<=1，eq表示等式约束；ineq表示大于等于0约束
    cons = [{"type": "eq", "fun": lambda beta:1-beta[0]-beta[1]-beta[2]-beta[3]-beta[4] -beta[5]}]
    # 参数的初始迭代值
    x0 = np.array([ 0, 0, 0,0, 0, 0])
    # 最优化求解
    res = minimize(objfunc, args=(Y, ff), x0=x0,
                bounds=bnds, constraints=cons)
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
    Y = lf['基金日收益率']  # 因变量
    # 使用最小二乘逼近求解，定义目标函数，minimize误差平方和
    # 多元线性模型中含常数项，待估参数分别为beta0，beta1，beta2，beta3....
    def fun(beta, lf): return   beta[0]*lf['农林牧渔pct']+beta[1]*lf['基础化工pct']+beta[2]*lf['钢铁pct']\
                                +beta[3]*lf['有色金属pct']\
                                +beta[4]*lf['电子pct']+beta[5]*lf['家用电器pct']\
                                +beta[6]*lf['食品饮料pct']+beta[7]*lf['纺织服饰pct']+beta[8]*lf['轻工制造pct']\
                                +beta[9]*lf['医药生物pct']+beta[10]*lf['公用事业pct']\
                                +beta[11]*lf['交通运输pct']+beta[12]*lf['房地产pct']+beta[13]*lf['商贸零售pct']\
                                +beta[14]*lf['社会服务pct']+beta[15]*lf['综合pct']+beta[16]*lf['建筑材料pct']\
                                +beta[17]*lf['建筑装饰pct']+beta[18]*lf['电力设备pct']+beta[19]*lf['国防军工pct']\
                                +beta[20]*lf['计算机pct']+beta[21]*lf['传媒pct']\
                                +beta[22]*lf['通信pct']+beta[23]*lf['银行pct']+beta[24]*lf['非银金融pct']\
                                +beta[25]*lf['汽车pct']+beta[26]*lf['机械设备pct']+beta[27]*lf['煤炭pct']\
                                +beta[28]*lf['石油石化pct']+beta[29]*lf['环保pct']+beta[30]*lf['美容护理pct']
    def objfunc(beta, Y, lf): return np.sum(np.abs(Y - fun(beta, lf)) ** 2)#+lam*sum([pow(beta[n], 2) for n in range(1,4)])
    # 输入变量的边界条件,自变量的回归系数取值在（0，1）
    bnds=tuple([(0,1)]*31)
    # 设置约束条件，回归系数之和<=1，eq表示等式约束；ineq表示大于等于0约束
    def cons():
        cons= [{"type": "eq", "fun": lambda beta: 1 -(beta[0]+beta[1]+beta[2]+beta[3]+beta[4]+beta[5]+beta[6]+beta[7]+beta[8]+beta[9]+beta[10]+beta[11]\
                                                        +beta[12]+beta[13]+beta[14]+beta[15]+beta[16]+beta[17]+beta[18]+beta[19]+beta[20]+beta[21]+beta[22]\
                                                        +beta[23]+beta[24]+beta[25]+beta[26]+beta[27]+beta[28]+beta[29]+beta[30])}
            ]
        return cons
    cons=cons()
    # 参数的初始迭代值
    x0 = np.array([0]*31)
    # 最优化求解
    res = minimize(objfunc, args=(Y, lf),
                x0=x0, bounds=bnds, constraints=cons)
    ph=dict()
    for i,j in zip(index_name,range(31)):
            ph[i]=res.x[j]
    #ph['中债财富总值']=res.x[6]
    ph1=pd.DataFrame([ph])

    return ph1



st.title('基金风格及行业变化预测追踪 :blue[!] :sunglasses:')
code=st.text_input('请输入基金代码例如000001')
text1='**基于基金收益率威廉·夏普风格分析，实现对基金风格的高频跟踪**'
st.caption(text1)
text2='晨星风格箱法往往会考虑利用重仓股对基金风格进行分析，但股票基金每季度更新上一季度重仓股情况，频率较低且具有一定的滞后性.\
       同时当重仓股权重不高时，估计的基金风格代表性不足，此时可以考虑利用基金收益率对基金风格进行判断，以增强结果的时效性与代表性。\
       威廉·夏普于 1992 年提出基于收益率的风格分析法，主要思想是设立一系列风格指数，利用最优化的方法，\
       最小化基金收益率与系列风格指数收益率的残差平方和，得到股票基金相对于各风格指数的暴露.'
st.caption(text2)


if (code)and(st.button('开始')):
    with st.spinner('正在获取数据...'):
        fund_df = ak.fund_open_fund_info_em(fund=code, indicator="累计净值走势")
        fund_df['基金日收益率']=fund_df['累计净值'].pct_change().fillna(0)
        fund_df['净值日期']=pd.to_datetime(fund_df['净值日期'])
        fund_df=fund_df.rename(columns={'净值日期':'date'}).set_index('date')
    if len(fund_df)>10:
        st.success('成功获取数据!')
    else:
        st.write('请输入正确的基金代码')
    
    with st.spinner('正在运行请稍候...'):

        f_new=pd.merge(fund_df,style_index,left_index=True,right_index=True,how='left').dropna()

        df=Parallel(n_jobs=4)(delayed(size_analy)(f_new,year_month[i+1],year_month[i]) for i in range(6))

        re_df=pd.concat(df,ignore_index=True)
        re_df=re_df.apply(lambda x: x*100)
        re_df['日期']=year_month[:-1]

        fi=re_df[['日期','大盘成长','大盘价值','中盘成长','中盘价值','小盘成长','小盘价值']]
        #fi=fi[['大盘成长','大盘价值','中盘成长','中盘价值','小盘成长','小盘价值']].apply(lambda x: x*100)

        sw1_pct.index=pd.DatetimeIndex(sw1_pct.index)
        f_sw=pd.merge(sw1_pct,fund_df,left_index=True,right_index=True,how='left').dropna()

        df1=Parallel(n_jobs=6)(delayed(size_sw)(f_sw,year_month[i+1],year_month[i]) for i in range(6))
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

        

        st.dataframe(fi)

        st.plotly_chart(fig)

        st.plotly_chart(fig2)

        if st.checkbox('展示行业变化数据'):
            st.dataframe(fi_new)

