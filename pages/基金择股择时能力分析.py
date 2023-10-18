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
st.markdown("# 基金择股择时能力分析")
st.sidebar.header("基金择股择时能力分析")
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
index=st.text_input('请输入指数代码例如000300.SH')
if index:
    df = pro.fund_nav(ts_code=code,start_date=开始,end_date=结束,fields=['ts_code','nav_date','accum_nav'])
    df8=df.sort_values(by='nav_date',ignore_index=True)
    df2=df8.rename(columns={'nav_date':'time'})
    df0=df2[['time','accum_nav']]
    df0['pre_cumulative_nav'] = df0['accum_nav'].shift(1)
    df9=df0.set_index('time')
    df1 = df9[开始:结束]
    #df_one['fundsname'] = df_fund_info.fundsname.values[0]
    dfp=df = pro.index_daily(ts_code='000300.SH', start_date=开始, end_date=结束,fields=['ts_code','trade_date','close'])
    dfg=dfp.rename(columns={'trade_date':'time'})
    dfg1=dfg.sort_values(by='time',ignore_index=True)
    df_index_day=dfg1.set_index('time')
    nav_one = pd.merge(df1, df_index_day, left_index=True,
                            right_index=True, how='left')
    nav_one['cumulative_nav_pct'] = nav_one['accum_nav'].pct_change()
    nav_one['close_pct'] = nav_one['close'].pct_change()
    nav_df_part=nav_one
#if len(nav_df_part['accum_nav'])>2:
    adj_nav_end = list(nav_df_part['accum_nav'])[-1]  # 复权累计净值的区间末位数值
    adj_nav_start = list(nav_df_part['accum_nav'])[0]  # 复权累计净值的区间首位数值
    nav_shift1_start = list(nav_df_part['pre_cumulative_nav'])[0]
# 复权累计净值归一、指数收盘价归一,待后续使用
    nav_df_part['nav_unit'] = nav_df_part['accum_nav'] / adj_nav_start

    nav_df_part['close_unit'] = nav_df_part['close'] / \
        list(nav_df_part['close'])[0]
    # 样本期的绝对收益率
    abs_ret = (adj_nav_end/adj_nav_start)-1
        #样本期的年化收益率
    annual_ret = pow(adj_nav_end/adj_nav_start, 250/(len(nav_df_part)-1))-1

    #计算胜率
    fenmu=len(nav_df_part)
    sd=nav_df_part.loc[nav_df_part['cumulative_nav_pct']>0]
    fenzi=len(sd)
    victory_days=fenzi/fenmu
    #样本期的最大回撤
    #nav_df_part=nav_one
    interval_max_down = ((nav_df_part['accum_nav'].cummax()-nav_df_part['accum_nav']) /
                        (nav_df_part['accum_nav'].cummax())).max()

    # 样本期年化波动率
    
    annual_var = nav_df_part['cumulative_nav_pct'].std(
            ddof=1)*pow(250, 0.5)

    # 样本期间年化夏普，年化后的平均收益率-无风险利率 /年化后的波动率
    rf_rate=0.02
    annual_sharpe = (
            pow((1+nav_df_part['cumulative_nav_pct'].mean()), 250)-1-rf_rate)/annual_var
    #计算卡玛比率
    interval_calmar = annual_ret/interval_max_down

    # 样本期下行波动率
    temp = nav_df_part[nav_df_part['cumulative_nav_pct']
                        < nav_df_part['cumulative_nav_pct'].mean()]
    temp2 = temp['cumulative_nav_pct'] - \
            nav_df_part['cumulative_nav_pct'].mean()
    down_var = np.sqrt((temp2**2).sum()/(len(nav_df_part)-1))*pow(250, 0.5)
    df_a_t = nav_df_part
    rf_rate=0.02
    df_a_t['rf'] = rf_rate/250  # 日度
                # 以上数据整合完毕

                # 完善模型自变量和因变量数据
    df_a_t['rprf'] = df_a_t['cumulative_nav_pct'] - df_a_t['rf']  # 因变量rp-rf
    df_a_t['rmrf'] = df_a_t['close_pct'] - df_a_t['rf']  # 自变量rm-rf
    df_a_t['rmrf2'] = df_a_t['rmrf'] * df_a_t['rmrf']  # 自变量rmrf^2
    df_a_t['cons'] = 1  # 常数项

                # 使用T-M模型计算选股收益、择时能力
    regmodel = OLS(df_a_t.rprf, df_a_t[[
                    'cons', 'rmrf', 'rmrf2']], missing='drop', hasconst=True).fit()  # 注意需要有常数项
    alpha = regmodel.params['cons']*250  # 简单年化选股收益
    timing = regmodel.params['rmrf2']  # 择时能力的回归系数
    ptiming = (1 if regmodel.pvalues['rmrf2']
                        <= 0.1 else 0)  # 择时能力是否显著 0不显著 1显著
    r2 = regmodel.rsquared_adj  # 调整后的R2

                # 使用CAPM模型计算超额收益
    regmodel2 = OLS(
                    df_a_t.rprf, df_a_t[['cons', 'rmrf']], missing='drop', hasconst=True).fit()
    exaplpha = regmodel2.params['cons']*250  # CAPM超额收益
    timing_y = exaplpha-alpha  # 择时收益=CAPM的超额收益-TM选股收益

                # 运气还是实力通过BootStrap重复抽样计算alpha
                # 首先取出CAPM模型中的残差序列、拟合值
    dfres = pd.DataFrame({'fit': regmodel2.fittedvalues,
                                    'resid': regmodel2.resid})  # 取出CAPM模型的回归拟合值和残差项
    df_alpha_res = pd.merge(df_a_t, dfres, right_index=True,
                                        left_index=True, how='inner')  # 整合原始数据和拟合数据
    df_alpha_res = df_alpha_res.copy()
                # 计算伪净值  y_hat=b1*(rm-rf)+resid,没有截距项
    df_alpha_res['fit_hat'] = regmodel2.params['rmrf'] * \
                    df_alpha_res['rmrf'] + df_alpha_res['resid']
    df_alpha_res = df_alpha_res.reset_index()  # 方便获取索引
    num = len(df_alpha_res)  # 样本的个数

    sample_mean_list = []
    for i in range(1000):  # bootstrap 1000次
        index = np.random.choice(range(num), num)  # 有放回抽样
        df_alpha_res_sample = df_alpha_res.iloc[index]
        reg_sample = OLS(df_alpha_res_sample.fit_hat, df_alpha_res_sample[['cons', 'rmrf']],
                                    missing='drop', hasconst=True).fit()
                    
        alpha_sample = reg_sample.params['cons'] * 250  # 伪净值的年化选股收益
        sample_mean_list.append(alpha_sample)
                # plt.hist(sample_mean_list)                                             #查看分布
    ser = pd.Series(sample_mean_list)
    p_alpha = len(ser[ser > exaplpha])/len(ser)  # 简单计算拒绝域的概率
                # aplha是否是运气还是实力 0为运气 1为实力
    p_alpha = (1 if p_alpha <= 0.1 else 0)
    basic_factor_dict = {'绝对收益率': abs_ret, '年化收益率': annual_ret,
                            '区间最大回撤': interval_max_down, '年化波动率': annual_var,
                            '年化夏普': annual_sharpe, '卡玛比率': interval_calmar,
                            '下行波动率': down_var,'胜率':victory_days}
    al_ti = {'绝对收益率':basic_factor_dict['绝对收益率'], "年化选股收益": alpha, 'TM模型调整后R2': r2, "择时系数": timing,
                            "择时显著性": ptiming, "年化择时收益": timing_y, "alpha实力": p_alpha}
    st.dataframe(al_ti)


