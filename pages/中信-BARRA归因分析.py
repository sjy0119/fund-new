from scipy.optimize import minimize
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import akshare as ak
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] =False 

st.set_page_config(page_icon="😎",)

st.markdown("# 中信-BARRA归因分析")
st.sidebar.header("中信-BARRA归因分析")
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
#缓存指数数据
@st.cache_data(ttl=300)
def load_data_index():
    df_hist1=pd.read_csv("指数数据.csv")
    return df_hist1
df_hist2=load_data_index()

#缓存barra日收益率数据
@st.cache_data(ttl=300)
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
barra_factor1=load_barra_data()

start_date = st.date_input(
    "请选择开始日期",
    date(2020,2,9))
#st.write('开始日期:', start_date)
start=str(start_date)
end_date = st.date_input(
    "请选择结束日期",
    date(2021,5,9))
#st.write('结束日期:',end_date)
end=str(end_date)


code=st.text_input('请输入基金代码例如000001')
index=st.selectbox("请选择比较基准",
   ("沪深300", "中证500", "中证800",'中证1000','上证50','科创50'))

if index=='沪深300':
    index='000300'
elif index=='中证500':
    index='000905'
elif index=='上证50':
    index='000016'
elif index=='中证1000':
    index='000852'
elif index=='中证800':
    index='000906'
elif index=='科创50':
    index='000608'

if (index=='000300')|(index=='000905')|(index=='000906')|(index=='000852')|(index=='000016')|(index=='000688'):
    index='sh'+index
#指数列表 
wind_index = df_hist2[['tradedate']+CIStyleindex_list]
wind_index['tradedate'] =pd.to_datetime( wind_index['tradedate'])
wind_index=wind_index.rename(columns={'tradedate':'date'})
wind_index=wind_index.set_index('date')
for i in wind_index.columns:
    wind_index[i+'pct']=wind_index[i].pct_change()

def size_CIS(df):
    """
    定义一个计算基金风格的函数
    """
    lf=df.copy()
    #lf=lf[start_date:end_date]
    b1 = np.array(lf[code+'pct'])  # 因变量
    A1 = np.array(lf[['cons','CI005921.WIpct','CI005920.WIpct','CI005919.WIpct','CI005918.WIpct','CI005917.WIpct']])
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
    x0 = np.array([-100, 0, 0, 0, 0, 0])
    bnds = [(None,None),(0,1)]
    for i in range(num_x-2):
        bnds.append((0,1))
    res = minimize(my_func, 
                bounds = bnds, x0=x0,
                constraints=cons)
    
    ph=dict()
    for i,j in zip(['稳定风格','成长风格','消费风格','周期风格','金融风格'],range(1,6)):
            ph[i]=res.x[j]
    ph1=pd.DataFrame([ph])

    return ph1,res

def cal_relative(df):
    """
    定义一个计算相对风格的函数
    """
    lf=df.copy()
    #lf=lf[start_date:end_date]
    b1 = np.array(lf[code+'pct'])  # 因变量
    A1 = np.array(lf[['cons','CI005921.WIpct','CI005920.WIpct','CI005919.WIpct','CI005918.WIpct','CI005917.WIpct']])
    b2 = np.array(lf['close_pct'])
    def minmean(A1,b1):
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
        x0 = np.array([-100, 0, 0, 0, 0, 0])
        bnds = [(None,None),(0,1)]
        for i in range(num_x-2):
            bnds.append((0,1))
        res = minimize(my_func, 
                    bounds = bnds, x0=x0,
                    constraints=cons)
        return res
    res1=minmean(A1,b1)
    res2=minmean(A1,b2)
    #'稳定风格','成长风格','消费风格','周期风格','金融风格'
    beta_fund0 = res1.x[0]  # 常数项
    beta_fund1 = res1.x[1]  # 稳定风格
    beta_fund2 = res1.x[2]  # 成长风格
    beta_fund3 = res1.x[3]  # 消费风格
    beta_fund4 = res1.x[4]  # 周期风格
    beta_fund5 = res1.x[5]  # 金融风格

    beta_index0 = res2.x[0]  # 常数项
    beta_index1 = res2.x[1]  # 稳定风格
    beta_index2 = res2.x[2]  # 成长风格
    beta_index3 = res2.x[3]  # 消费风格
    beta_index4 = res2.x[4]  # 周期风格
    beta_index5 = res2.x[5]  # 金融风格
    # 相对风格暴露为
    jrfg = beta_fund1 - beta_index1
    zqfg = beta_fund2 - beta_index2
    xffg = beta_fund3 - beta_index3
    czfg = beta_fund4 - beta_index4
    wdfg = beta_fund5 - beta_index5

    CIS_guiying=dict()
    CIS_guiying[ '稳定风格相对暴露'] = jrfg
    CIS_guiying[ '成长风格相对暴露'] = zqfg
    CIS_guiying[ '消费风格相对暴露'] = xffg
    CIS_guiying[ '周期风格相对暴露'] = czfg
    CIS_guiying[ '金融风格相对暴露'] = wdfg

    CIS_guiying1=pd.DataFrame([CIS_guiying])

    return CIS_guiying1

barra_factor_list =['BarraCNE5_Beta_nav_pct', 'BarraCNE5_BooktoPrice_nav_pct', 'BarraCNE5_DebttoAssets_nav_pct',
    'BarraCNE5_EarningsYield_nav_pct', 'BarraCNE5_Growth_nav_pct', 'BarraCNE5_Liquidity_nav_pct',
    'BarraCNE5_Momentum_nav_pct', 'BarraCNE5_NonLinearSize_nav_pct',
    'BarraCNE5_ResidualVolatility_nav_pct', 'BarraCNE5_Size_nav_pct']

def barra_ana(df):
    lf=df.copy()
    lam = 6*pow(10, -5)  # 正则惩罚项
    b1 = np.array(lf[code+'pct'])  # 因变量
    A1 = np.array(lf[['cons','close_pct','CI005921.WIpct','CI005920.WIpct','CI005919.WIpct','CI005918.WIpct','CI005917.WIpct']+barra_factor_list])
    def minmean(A1,b1):
        num_x = np.shape(A1)[1]
        def my_func(x):
            ls = np.abs((b1-np.dot(A1,x))**2)
            ld=lam*np.sum([pow(x[n], 2) for n in range(7, 17)])
            result = np.sum(ls)+ld
            return result
        def g1(x):
            return np.sum(x) #sum of X >= 0
        def g2(x):
            return 1-np.sum(x) #sum of X = 1
        cons = ({'type': 'ineq', 'fun': g1}
                ,{'type': 'eq', 'fun':  g2})
        x0  = np.array([-100, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -100, -100, -100, -
                100, -100, -100, -100, -100, -100, -100])
        bnds = [(None, None), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (None, None), (None, None), (None, None), (None, None), (None, None),
            (None, None), (None, None), (None, None), (None, None), (None, None)]
        res = minimize(my_func, 
                    bounds = bnds, x0=x0,
                    constraints=cons)
        return res
    res=minmean(A1,b1)
    beta_cons = res.x[0]  # 常数项
    beta_fm = res.x[1]
    beta_5917 = res.x[2]
    beta_5918 = res.x[3]
    beta_5919 = res.x[4]
    beta_5920 = res.x[5]
    beta_5921 = res.x[6]
    beta_beta = res.x[7]
    beta_book_to_price = res.x[8]
    beta_earning_yield = res.x[9]
    beta_growth = res.x[10]
    beta_leverage = res.x[11]
    beta_liquidity = res.x[12]
    beta_momentum = res.x[13]
    beta_non_linear_size = res.x[14]
    beta_residual_volatility = res.x[15]
    beta_size = res.x[16]
     #'稳定风格','成长风格','消费风格','周期风格','金融风格'
    CIS_Barra =dict()
    CIS_Barra['选股收益'] = beta_cons
    CIS_Barra['贝塔'] = beta_fm
    CIS_Barra['稳定风格暴露'] = beta_5917
    CIS_Barra['成长风格暴露'] = beta_5918
    CIS_Barra['消费风格暴露'] = beta_5919
    CIS_Barra['周期风格暴露'] = beta_5920
    CIS_Barra['金融风格暴露'] = beta_5921
    CIS_Barra['Beta因子暴露'] = beta_beta
    CIS_Barra['账面市值比因子暴露'] = beta_book_to_price
    CIS_Barra['盈利预期因子暴露'] = beta_earning_yield
    CIS_Barra['成长因子暴露'] = beta_growth
    CIS_Barra['杠杆因子暴露'] = beta_leverage
    CIS_Barra['流动性因子暴露'] = beta_liquidity
    CIS_Barra['动量因子暴露'] = beta_momentum
    CIS_Barra['非线性市值因子暴露'] = beta_non_linear_size
    CIS_Barra['残差波动率因子暴露'] = beta_residual_volatility
    CIS_Barra['市值因子暴露'] = beta_size
    CIS_Barra=pd.DataFrame([CIS_Barra])
    return CIS_Barra,res


@st.cache_data(ttl=300)
def load_data(code,index):
    fund_nav = ak.fund_open_fund_info_em(fund=code, indicator="累计净值走势").rename(columns={'净值日期':'date','累计净值':code})
    fund_nav['date']=pd.to_datetime(fund_nav['date'])
    sh300 = ak.stock_zh_index_daily(symbol=index)[['date','close']]
    sh300['date']=pd.to_datetime(sh300['date'])
    df=pd.merge(fund_nav,sh300,on='date',how='inner')
    df[f'{code}pct']=df[code].pct_change().fillna(0)
    df['close_pct']=df['close'].pct_change().fillna(0)
    df=df.set_index('date')
    return df

if code:
    fund_df=load_data(code,index)

    all_df1=pd.merge(fund_df,wind_index,left_index=True,right_index=True,how='left').dropna()
    all_df=pd.merge(all_df1,barra_factor1,left_index=True,right_index=True,how='left').dropna()
    all_df['cons']=1
   
    df_list=[all_df['2023-09':'2023-10'],all_df['2023-08':'2023-09'],all_df['2023-07':'2023-08'],all_df['2023-06':'2023-07'],all_df['2023-05':'2023-06']
             ,all_df
             ]
   
    cis_all_df=[size_CIS(df)[0] for df in df_list]

    cis_relative=[cal_relative(df) for df in df_list]

    cis_barra=[barra_ana(df)[0] for df in df_list]

    df1=pd.concat(cis_all_df)
    df1.insert(0,'时间区间',['2023-09:2023-10','2023-08:2023-09','2023-07:2023-08','2023-06:2023-07','2023-05:2023-06','成立以来'])

    df2=pd.concat(cis_relative)
    df2.insert(0,'时间区间',['2023-09:2023-10','2023-08:2023-09','2023-07:2023-08','2023-06:2023-07','2023-05:2023-06','成立以来'])

    df3=pd.concat(cis_barra)
    df3.insert(0,'时间区间',['2023-09:2023-10','2023-08:2023-09','2023-07:2023-08','2023-06:2023-07','2023-05:2023-06','成立以来'])

    if end_date>start_date:

        all_df2=all_df[start:end]
        op=barra_ana(all_df2)
        res1=op[1]
        CIS_Barra= op[0]
        all_df2['因子贡献收益_pct']=list(np.dot(np.array(all_df2[['cons','close_pct','CI005921.WIpct','CI005920.WIpct','CI005919.WIpct','CI005918.WIpct','CI005917.WIpct']+barra_factor_list])
                                        ,np.array([res1.x[i] for i in range(17)])))
        all_df2['nav_yield'] = 1
        all_df2['nav_yield'].iloc[1:] = (
                1+all_df2[code+'pct'].iloc[1:]).cumprod()
        all_df2['CISty_barra_fit_yield'] = 1
        all_df2['CISty_barra_fit_yield'].iloc[1:] = (
                1+all_df2['因子贡献收益_pct'].iloc[1:]).cumprod()
        all_df2['选股收益贡献'] =list(np.dot(np.array(all_df2['cons']),res1.x[0]))

        all_df2['市场收益贡献'] = list(np.dot(np.array(all_df2['close_pct']),res1.x[1]))

        all_df2['风格因子收益贡献'] = list(np.dot(np.array(all_df2[['CI005921.WIpct','CI005920.WIpct','CI005919.WIpct','CI005918.WIpct','CI005917.WIpct']])
                                                ,np.array([res1.x[i] for i in range(2,7)])))
        all_df2['Barra因子收益贡献']= list(np.dot(np.array(all_df2[barra_factor_list]),np.array([res1.x[i] for i in range(7,17)])))
        
        all_df2['特质收益率贡献'] = all_df2[code+'pct']-all_df2['选股收益贡献'] - \
                all_df2['市场收益贡献']-all_df2['风格因子收益贡献'] - \
                all_df2['Barra因子收益贡献']
        all_df2.index = pd.to_datetime(all_df2.index)
        
        nav_yiled_trace_v2 = go.Scatter(x=all_df2.index.strftime('%Y/%m/%d'),
                                        y=all_df2['nav_yield'], mode='lines', name=code)
        CIStyfit_yield_trace_v2 = go.Scatter(x=all_df2.index.strftime('%Y/%m/%d'),
                                            y=all_df2['CISty_barra_fit_yield'], mode='lines', name='风格因子贡献收益')
        fig_nav_CIS_barra = go.Figure(
            data=[nav_yiled_trace_v2, CIStyfit_yield_trace_v2])

        fig_nav_CIS_barra.update_layout(
            title_text="基金收益与风格因子贡献收益 <br> 最新净值日期:" +
            all_df2.index[-1].strftime('%Y-%m-%d'),
            margin=dict(l=100, r=100, t=60, b=80),
            yaxis={'tickformat': '.2f', 'title': ' 净值'},
            xaxis={'tickangle': -70, 'tickformat': '%Y-%m-%d'})
        st.plotly_chart(fig_nav_CIS_barra)
        
        Barra_expose = CIS_Barra[['Beta因子暴露', '账面市值比因子暴露', '盈利预期因子暴露', '成长因子暴露',
                                '杠杆因子暴露', '流动性因子暴露', '动量因子暴露', '非线性市值因子暴露', '残差波动率因子暴露', '市值因子暴露']]
        Barra_expose = Barra_expose.T
        Barra_expose_trace = go.Bar(
            x=Barra_expose.index.to_list(), y=Barra_expose.iloc[:, 0].to_list())
        layout = go.Layout(
            title='Barra因子暴露 <br> 当前日期为{}'.format(
                datetime.now().strftime('%Y-%m-%d')),
            yaxis={'title': '因子暴露', 'tickformat': '.2f'},
            xaxis={'title': 'BarraCne5因子'})
        Barra_expose_bar = go.Figure(data=Barra_expose_trace, layout=layout)
        st.plotly_chart(Barra_expose_bar)

        barra_factor2_list =['BarraCNE5_Beta_nav', 'BarraCNE5_BooktoPrice_nav', 'BarraCNE5_DebttoAssets_nav',
        'BarraCNE5_EarningsYield_nav', 'BarraCNE5_Growth_nav', 'BarraCNE5_Liquidity_nav',
        'BarraCNE5_Momentum_nav', 'BarraCNE5_NonLinearSize_nav',
        'BarraCNE5_ResidualVolatility_nav', 'BarraCNE5_Size_nav']

        list1=[all_df2[i][-1]/all_df2[i][0]-1 for i in barra_factor2_list]

        fig1= go.Figure(data = (go.Bar(x=['Beta因子暴露', '账面市值比因子暴露', '盈利预期因子暴露', '成长因子暴露',
                                '杠杆因子暴露', '流动性因子暴露', '动量因子暴露', '非线性市值因子暴露', '残差波动率因子暴露', '市值因子暴露'],  
                                        y=list1 )))
        fig1.update_layout(title_text='区间风格因子收益')
        st.plotly_chart(fig1)

        # 绘制因子收益贡献分解图 
        yield_decomposition_trace1 = go.Bar(x=all_df2.index.strftime('%Y/%m/%d'), y=all_df2['选股收益贡献'], name='选股收益贡献')
        yield_decomposition_trace2 = go.Bar(x=all_df2.index.strftime('%Y/%m/%d'), y=all_df2['市场收益贡献'], name='市场收益贡献')
        yield_decomposition_trace3 = go.Bar(x=all_df2.index.strftime('%Y/%m/%d'), y=all_df2['风格因子收益贡献'], name='风格因子收益贡献')
        yield_decomposition_trace4 = go.Bar(x=all_df2.index.strftime('%Y/%m/%d'), y=all_df2['Barra因子收益贡献'], name='Barra因子收益贡献')
        yield_decomposition_trace5 = go.Bar(x=all_df2.index.strftime('%Y/%m/%d'), y=all_df2['特质收益率贡献'], name='特质收益率贡献')

        layout = go.Layout(
            title='基金收益分解 <br> 当前日期为{}'.format(
                datetime.now().strftime('%Y-%m-%d')),
            yaxis={'title': '收益率', 'tickformat': '.2%'},
            xaxis={'title': '日期'}, barmode='stack')
        yield_decomposition_bar = go.Figure(data=[yield_decomposition_trace1, yield_decomposition_trace2,
                                                yield_decomposition_trace3, yield_decomposition_trace4, yield_decomposition_trace5], layout=layout)
        st.plotly_chart(yield_decomposition_bar)

    st.dataframe(df3[['时间区间']+['Beta因子暴露', '账面市值比因子暴露', '盈利预期因子暴露', '成长因子暴露',
                            '杠杆因子暴露', '流动性因子暴露', '动量因子暴露', '非线性市值因子暴露', '残差波动率因子暴露', '市值因子暴露']],hide_index=True)
