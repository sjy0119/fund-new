from scipy.optimize import minimize
from plotly.offline import iplot, init_notebook_mode
from statsmodels.regression.linear_model import OLS  # 线性回归
import plotly as py
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import seaborn as sns
import tushare as ts
pro = ts.pro_api('8e812052c92d7a829f0e3b0197d248e48bb2ba3efbbaa60f505e6852')
import holoviews as hv
import streamlit as st
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
from datetime import datetime, time, timedelta,date
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
@st.cache_data
def load_data():
    df_hist1=pd.read_csv(r"C:\Users\WuKangmin\Desktop\基金数据分析及Web可视化\指数数据.csv")
    #df_hist1=df_hist.rename(columns={'time':'tradedate'})
    return df_hist1
df_hist2=load_data()
#中信风格指数归因
#st.title('中信风格指数归因分析')
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
index=st.text_input('请输入指数代码例如000300.SH')
#指数列表 
wind_index = df_hist2[['tradedate']+CIStyleindex_list]
wind_index['tradedate'] =pd.to_datetime( wind_index['tradedate'])
wind_index=wind_index.set_index('tradedate')
wind_index_part = wind_index[开始:结束]  # 区间参数
#获取基金净值信息以及指数数据
if index:
    df = pro.fund_nav(ts_code=code,start_date=开始,end_date=结束,fields=['ts_code','nav_date','accum_nav'])
    #df['nav_date']=pd.to_datetime(df['nav_date'])
    df8=df.sort_values(by='nav_date',ignore_index=True)
    df2=df8.rename(columns={'nav_date':'tradedate'})
    df2['tradedate']=pd.to_datetime(df2['tradedate'])
    df0=df2[['tradedate','accum_nav']]
    df0['pre_cumulative_nav'] = df0['accum_nav'].shift(1)
    df9=df0.set_index('tradedate')
    df1 = df9[开始:结束]
    #df_one['fundsname'] = df_fund_info.fundsname.values[0]
    dfp=df = pro.index_daily(ts_code=index, start_date=开始, end_date=结束,fields=['ts_code','trade_date','close'])
    dfg=dfp.rename(columns={'trade_date':'tradedate'})
    dfg['tradedate']=pd.to_datetime(dfg['tradedate'])
    dfg['close_pct']=dfg['close'].pct_change()
    #dfg['tradedate'] = pd.to_datetime(dfg['tradedate'])
    dfg1=dfg.sort_values(by='tradedate',ignore_index=True)
    df_index_day=dfg1.set_index('tradedate')
    nav_one = pd.merge(df1, df_index_day, left_index=True,
                            right_index=True, how='left')
    nav_part_left1=nav_one[开始:结束]
    nav_CISty = pd.merge(nav_part_left1, wind_index_part,
                        left_index=True, right_index=True, how='left')
    for i in CIStyleindex_list:
        nav_CISty[i+'pct'] = nav_CISty[i].pct_change()
    nav_CISty['cons'] = 1  # 添加常数项
    nav_CISty['refactor_net_value_pct']=nav_CISty['accum_nav'].pct_change()
    # 计算风格暴露,产品的周收益率对风格指数收益率回归
    Y = nav_CISty['refactor_net_value_pct']  # 因变量
    # 使用最小二乘逼近求解，定义目标函数，minimize误差平方和
    # 多元线性模型中含常数项，待估参数分别为beta0，beta1，beta2，beta3....

    def fun(beta, nav_CISty): return beta[0] * nav_CISty['cons'] + beta[1] * nav_CISty['CI005917.WIpct'] + beta[2] * \
        nav_CISty['CI005918.WIpct'] + beta[3] * nav_CISty['CI005919.WIpct'] + \
        beta[4] * nav_CISty['CI005920.WIpct'] + \
        beta[5] * nav_CISty['CI005921.WIpct']

    def objfunc(beta, Y, nav_CISty): return np.sum(np.abs(Y - fun(beta, nav_CISty)) ** 2)

    # 输入变量的边界条件,自变量的回归系数取值在（0，1）
    bnds = [(None, None), [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
    # 设置约束条件，回归系数之和<=1，eq表示等式约束；ineq表示大于等于0约束
    cons = [{"type": "ineq", "fun": lambda beta: 1 -
            beta[1] - beta[2] - beta[3] - beta[4] - beta[5]}]
    # 参数的初始迭代值
    x0 = np.array([-100, 0, 0, 0, 0, 0])
    # 最优化求解
    res = minimize(objfunc, args=(Y, nav_CISty),
                x0=x0, bounds=bnds, constraints=cons)
    # result.x返回最优解,即风格暴露
    beta_fund0 = res.x[0]  # 常数项
    beta_fund1 = res.x[1]  # 金融风格
    beta_fund2 = res.x[2]  # 周期风格
    beta_fund3 = res.x[3]  # 消费风格
    beta_fund4 = res.x[4]  # 成长风格
    beta_fund5 = res.x[5]  # 稳定风格

    # 计算相对指数的的相对风格暴露
    # 计算风格暴露,产品的周收益率对风格指数收益率回归
    Y2 = nav_CISty['close_pct']  # 因变量

    def fun(beta, nav_CISty): return beta[0] * nav_CISty['cons'] + beta[1] * nav_CISty['CI005917.WIpct'] + beta[2] * \
        nav_CISty['CI005918.WIpct'] + beta[3] * nav_CISty['CI005919.WIpct'] + \
        beta[4] * nav_CISty['CI005920.WIpct'] + \
        beta[5] * nav_CISty['CI005921.WIpct']

    def objfunc(beta, Y2, nav_CISty): return np.sum(
        np.abs(Y2 - fun(beta, nav_CISty)) ** 2)

    # 最优化求解
    res2 = minimize(objfunc, args=(Y2, nav_CISty),
                    x0=x0, bounds=bnds, constraints=cons)
    # result.x返回最优解,即指数的风格暴露
    beta_index0 = res2.x[0]  # 常数项
    beta_index1 = res2.x[1]  # 金融风格
    beta_index2 = res2.x[2]  # 周期风格
    beta_index3 = res2.x[3]  # 消费风格
    beta_index4 = res2.x[4]  # 成长风格
    beta_index5 = res2.x[5]  # 稳定风格
    # 相对风格暴露为
    jrfg = beta_fund1 - beta_index1
    zqfg = beta_fund2 - beta_index2
    xffg = beta_fund3 - beta_index3
    czfg = beta_fund4 - beta_index4
    wdfg = beta_fund5 - beta_index5

    CIS_guiying=dict()
    CIS_guiying[ '金融风格暴露'] = beta_fund1
    CIS_guiying[ '周期风格暴露'] = beta_fund2
    CIS_guiying[ '消费风格暴露'] = beta_fund3
    CIS_guiying[ '成长风格暴露'] = beta_fund4
    CIS_guiying[ '稳定风格暴露'] = beta_fund5
    CIS_guiying[ '金融风格相对暴露'] = jrfg
    CIS_guiying[ '周期风格相对暴露'] = zqfg
    CIS_guiying[ '消费风格相对暴露'] = xffg
    CIS_guiying[ '成长风格相对暴露'] = czfg
    CIS_guiying[ '稳定风格相对暴露'] = wdfg

    # 绘制因子收益贡献曲线和产品收益贡献曲线

    nav_CISty['CISty_pct_fit'] = beta_fund0 * nav_CISty['cons'] + beta_fund1 * nav_CISty['CI005917.WIpct'] + beta_fund2 * \
            nav_CISty['CI005918.WIpct'] + beta_fund3 * nav_CISty['CI005919.WIpct'] + \
            beta_fund4 * nav_CISty['CI005920.WIpct'] + \
            beta_fund5 * nav_CISty['CI005921.WIpct']
    nav_CISty['nav_yield'] = 1
    nav_CISty['nav_yield'].iloc[1:] = (
            1+nav_CISty['refactor_net_value_pct'].iloc[1:]).cumprod()
    nav_CISty['CIStyfit_yield'] = 1
    nav_CISty['CIStyfit_yield'].iloc[1:] = (
            1+nav_CISty['CISty_pct_fit'].iloc[1:]).cumprod()
    nav_CISty.index=pd.to_datetime(nav_CISty.index)
    nav_yiled_trace = go.Scatter(x=nav_CISty.index.strftime('%Y/%m/%d'),
                                    y=nav_CISty['nav_yield'], mode='lines', name=code)
    CIStyfit_yield_trace = go.Scatter(x=nav_CISty.index.strftime('%Y/%m/%d'),
                                        y=nav_CISty['CIStyfit_yield'], mode='lines', name='风格因子贡献收益')
    fig_nav_CIS = go.Figure(data=[nav_yiled_trace, CIStyfit_yield_trace])

    fig_nav_CIS .update_layout(
            title_text="基金收益与中信风格因子贡献收益 <br> 最新净值日期:" +
            nav_CISty.index[-1].strftime('%Y-%m-%d'),
            margin=dict(l=100, r=100, t=60, b=80),
            yaxis={'tickformat': '.2f', 'title': ' 净值'},
            xaxis={'tickangle': -70, 'tickformat': '%Y-%m-%d'})
    st.plotly_chart(fig_nav_CIS )

    if st.checkbox('展示指标计算结果'):
        st.subheader('指标计算结果')
        st.dataframe(CIS_guiying)