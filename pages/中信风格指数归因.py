from scipy.optimize import minimize
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import akshare as ak
import plotly_express as px
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] =False #减号unicode编码
st.set_page_config(page_icon="😎",)

st.markdown("# 中信风格指数归因分析")
st.sidebar.header("中信风格指数归因分析")
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
def load_data():
    df_hist1=pd.read_csv("指数数据.csv")
    return df_hist1
df_hist2=load_data()


start_date = st.date_input(
    "请选择开始日期",
    date(2020,2,9))

strat=str(start_date)
end_date = st.date_input(
    "请选择结束日期",
    date(2021,5,9))
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
#获取基金净值信息以及指数数据

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
    b2 = np.array(lf[index+'pct'])
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


@st.cache_data(ttl=300)
def load_data(code,index):
    fund_nav = ak.fund_open_fund_info_em(fund=code, indicator="累计净值走势").rename(columns={'净值日期':'date','累计净值':code})
    fund_nav['date']=pd.to_datetime(fund_nav['date'])
    sh300 = ak.stock_zh_index_daily(symbol=index)[['date','close']].rename(columns={'close':index})
    sh300['date']=pd.to_datetime(sh300['date'])
    df=pd.merge(fund_nav,sh300,on='date',how='inner')
    df[f'{code}pct']=df[code].pct_change().fillna(0)
    df[index+'pct']=df[index].pct_change().fillna(0)
    df=df.set_index('date')
    return df

if code:
    fund_df=load_data(code,index)

    all_df=pd.merge(fund_df,wind_index,left_index=True,right_index=True,how='left').dropna()

    all_df['cons']=1
   
    df_list=[all_df['2023-09':'2023-10'],all_df['2023-08':'2023-09'],all_df['2023-07':'2023-08'],all_df['2023-06':'2023-07'],all_df['2023-05':'2023-06']
             ,all_df
             ]
    cis_all_df=[size_CIS(df)[0] for df in df_list]

    cis_relative=[cal_relative(df) for df in df_list]

    df1=pd.concat(cis_all_df)
    df1.insert(0,'时间区间',['2023-09:2023-10','2023-08:2023-09','2023-07:2023-08','2023-06:2023-07','2023-05:2023-06','成立以来'])

    df2=pd.concat(cis_relative)
    df2.insert(0,'时间区间',['2023-09:2023-10','2023-08:2023-09','2023-07:2023-08','2023-06:2023-07','2023-05:2023-06','成立以来'])


    # 计算风格暴露,产品的周收益率对风格指数收益率回归
    
    # 绘制因子收益贡献曲线和产品收益贡献曲线

    if end_date>start_date:
  
        all_df1=all_df[strat:end]
        res1=size_CIS(all_df1)[1]
        all_df1['CISty_pct_fit'] = list(np.array(all_df1[['cons','CI005921.WIpct','CI005920.WIpct','CI005919.WIpct','CI005918.WIpct','CI005917.WIpct']])@\
                                np.array([res1.x[i] for i in range(6)]))
        all_df1['nav_yield'] = 1
        all_df1['nav_yield'].iloc[1:] = (
                1+all_df1[code+'pct'].iloc[1:]).cumprod()
        all_df1['CIStyfit_yield'] = 1
        all_df1['CIStyfit_yield'].iloc[1:] = (
                1+all_df1['CISty_pct_fit'].iloc[1:]).cumprod()
        all_df1.index=pd.to_datetime(all_df1.index)
        nav_yiled_trace = go.Scatter(x=all_df1.index.strftime('%Y/%m/%d'),
                                        y=all_df1['nav_yield'], mode='lines', name=code)
        CIStyfit_yield_trace = go.Scatter(x=all_df1.index.strftime('%Y/%m/%d'),
                                            y=all_df1['CIStyfit_yield'], mode='lines', name='风格因子贡献收益')
        fig_nav_CIS = go.Figure(data=[nav_yiled_trace, CIStyfit_yield_trace])

        fig_nav_CIS .update_layout(
                title_text="基金收益与中信风格因子贡献收益 <br> 最新净值日期:" +
                all_df1.index[-1].strftime('%Y-%m-%d'),
                margin=dict(l=100, r=100, t=60, b=80),
                yaxis={'tickformat': '.2f', 'title': ' 净值'},
                xaxis={'tickangle': -70, 'tickformat': '%Y-%m-%d'})
        st.plotly_chart(fig_nav_CIS )

        fig = go.Figure(data = (go.Bar(x=['稳定风格','成长风格','消费风格','周期风格','金融风格'],  
                                       y=[res1.x[i] for i in range(1,6)] )))
        fig.update_layout(title_text='区间风格暴露')

        list1=[all_df1[i][-1]/all_df1[i][0]-1 for i in ['CI005921.WI','CI005920.WI','CI005919.WI','CI005918.WI','CI005917.WI']]

        fig1= go.Figure(data = (go.Bar(x=['稳定风格','成长风格','消费风格','周期风格','金融风格'],  
                                       y=list1 )))
        fig1.update_layout(title_text='区间风格因子收益')

        #st.caption('区间风格暴露')
        st.plotly_chart(fig)
  
        #st.caption('区间风格因子收益')
        st.plotly_chart(fig1)

    
    st.subheader('指标计算结果')
    st.caption('基金各区间风格')
    st.dataframe(df1)

    st.caption('基金相对于基准各区间风格')
    st.dataframe(df2)
