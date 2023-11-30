from statsmodels.regression.linear_model import OLS  # 线性回归
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta,date
import time
import akshare as ak
from dateutil.relativedelta import relativedelta

st.set_page_config(page_icon="😎",)
st.markdown("# 基金择股择时能力分析")
st.sidebar.header("基金择股择时能力分析")


@st.cache_data
def get_fund_name():
    df=pd.read_csv("股票基金",index_col=0)
    df['基金代码']=df['基金代码'].apply(lambda x: ('00000'+str(x))[-6:])
    return df
fund=get_fund_name()

fund_name=st.selectbox('请选择基金',tuple(fund['基金简称']))
code=fund.loc[fund['基金简称']==fund_name]['基金代码'].values[0]
index=st.selectbox("请选择基准",
   ("沪深300", "中证500", "中证800",'中证1000','上证50','科创50'))
st.caption('该模块为计算基金在近一年内四个时期的择股择时能力,由于数据是现爬现算,请耐心等待')

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

@st.cache_data(ttl=300)
def load_data(code,index):
    fund_nav = ak.fund_open_fund_info_em(fund=code, indicator="累计净值走势").rename(columns={'净值日期':'date','累计净值':'accum_nav'})
    fund_nav['date']=pd.to_datetime(fund_nav['date'])
    sh300 = ak.stock_zh_index_daily(symbol=index)[['date','close']]
    sh300['date'] = pd.to_datetime(sh300['date'])
    df=pd.merge(fund_nav,sh300,on='date',how='inner')
    df=df.set_index('date')
    return df


def cal_choose_stock_time(nav_one):
    nav_one['cumulative_nav_pct'] = nav_one['accum_nav'].pct_change()
    nav_one['close_pct'] = nav_one['close'].pct_change()
    nav_df_part=nav_one
#if len(nav_df_part['accum_nav'])>2:
    adj_nav_end = list(nav_df_part['accum_nav'])[-1]  # 复权累计净值的区间末位数值
    adj_nav_start = list(nav_df_part['accum_nav'])[0]  # 复权累计净值的区间首位数值
# 复权累计净值归一、指数收盘价归一,待后续使用
    abs_ret = (adj_nav_end/adj_nav_start)-1
        #样本期的年化收益率
    df_a_t = nav_df_part
    rf_rate=0.02
    df_a_t['rf'] = rf_rate/250  # 日度
                # 完善模型自变量和因变量数据
    df_a_t['rprf'] = list(np.array(df_a_t['cumulative_nav_pct']) - np.array(df_a_t['rf']))  # 因变量rp-rf
    df_a_t['rmrf'] = list(np.array(df_a_t['close_pct']) - np.array(df_a_t['rf']))  # 自变量rm-rf
    df_a_t['rmrf2'] = list(np.multiply(np.array(df_a_t['rmrf']),np.array(df_a_t['rmrf'])))  # 自变量rmrf^2
    df_a_t['cons'] = 1  # 常数项

                # 使用T-M模型计算选股收益、择时能力
    regmodel = OLS(df_a_t.rprf
                ,df_a_t[[
                    'cons', 'rmrf', 'rmrf2']]
                    , missing='drop', hasconst=True).fit()  # 注意需要有常数项
    alpha = regmodel.params['cons']*250  # 简单年化选股收益
    timing = regmodel.params['rmrf2']  # 择时能力的回归系数
    ptiming = (1 if regmodel.pvalues[-1]
                        <= 0.1 else 0)  # 择时能力是否显著 0不显著 1显著
    r2 = regmodel.rsquared_adj  # 调整后的R2

                # 使用CAPM模型计算超额收益
    regmodel2 = OLS(
                    df_a_t.rprf, df_a_t[['cons', 'rmrf']]
                    , missing='drop', hasconst=True).fit()
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
    df_alpha_res['fit_hat'] = regmodel2.params['rmrf']*df_alpha_res['rmrf'] + df_alpha_res['resid']
    df_alpha_res = df_alpha_res.reset_index()  # 方便获取索引
    num = len(df_alpha_res)  # 样本的个数

    #sample_mean_list = []
    #for i in range(1000):  # bootstrap 1000次
    def cal_for(df_alpha_res,num):
        index = np.random.choice(range(num), num)  # 有放回抽样
        df_alpha_res_sample = df_alpha_res.iloc[index]
        reg_sample = OLS(
            df_alpha_res_sample.fit_hat, df_alpha_res_sample[['cons', 'rmrf']]
            ,
                                    missing='drop', hasconst=True).fit()
        alpha_sample = reg_sample.params['cons'] * 250  # 伪净值的年化选股收益
        return alpha_sample
    sample_mean_list=[ cal_for(df_alpha_res,num) for _ in range(1000)]
                # plt.hist(sample_mean_list)                                             #查看分布
    ser = pd.Series(sample_mean_list)
    p_alpha = len(ser[ser > exaplpha])/len(ser)  # 简单计算拒绝域的概率
                # aplha是否是运气还是实力 0为运气 1为实力
    p_alpha = (1 if p_alpha <= 0.1 else 0)
    al_ti = {'绝对收益率':abs_ret, "年化选股收益": alpha, 'TM模型调整后R2': r2, "择时系数": timing,
                            "择时显著性": ptiming, "年化择时收益": timing_y, "alpha实力": p_alpha}
    al_ti=pd.DataFrame([al_ti])
    return al_ti
        #dat[f'{year_month[i+1]}:{year_month[i]}']=al_ti
@st.cache_data(ttl=300)
def get_data(all_df):
    rt=[cal_choose_stock_time(df) for df in all_df]
    return rt

today=date.today()

year_month=[str(today)[:-3]]+[str(date.today()-relativedelta(months=i+3))[:-3] for i in [0,3,6,9]]

if code:
    df=load_data(code,index)
    d=time.time()
    all_df=[df[year_month[i+1]:year_month[i]] for i in range(4)]
    rtf=get_data(all_df)
    #al_ti=pd.concat([rtf[f'{year_month[i+1]}:{year_month[i]}'] for i in range(4) ] )
    llop=[f'{year_month[i+1]}:{year_month[i]}' for i in range(4)]
    al_ti=pd.concat(rtf)
    al_ti.insert(0,'时间',llop)
    st.dataframe(al_ti,hide_index=True)
    d2=time.time()
    st.write(f'计算用时共:{round(d2-d,3)}秒')


