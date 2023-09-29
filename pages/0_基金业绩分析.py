# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta,date
from time import strftime 
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio
import tushare as ts
pro = ts.pro_api('8e812052c92d7a829f0e3b0197d248e48bb2ba3efbbaa60f505e6852')
import holoviews as hv

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] =False #减号unicode编码
st.set_page_config(page_icon="😎",)
st.markdown("# 基金业绩分析")
st.sidebar.header("基金业绩分析")
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
else:
    abs_ret=0
    annual_ret=0
    interval_max_down=0
    annual_var=0
    annual_sharpe=0
    interval_calmar=0
    down_var=0
    victory_days=0
# 结果字典
basic_factor_dict = {'绝对收益率': abs_ret, '年化收益率': annual_ret,
                        '区间最大回撤': interval_max_down, '年化波动率': annual_var,
                        '年化夏普': annual_sharpe, '卡玛比率': interval_calmar,
                        '下行波动率': down_var,'胜率':victory_days}

result_df = pd.DataFrame.from_dict({'分析结果': basic_factor_dict
                                    
                                    }, orient='index')

result_df.reset_index(inplace=True)
result_df['绝对收益率'] = result_df['绝对收益率'].map(lambda x: format(x, '.2%'))
result_df['年化收益率'] = result_df['年化收益率'].map(lambda x: format(x, '.2%'))
result_df['区间最大回撤'] = result_df['区间最大回撤'].map(
        lambda x: format(x, '.2%'))
result_df['年化波动率'] = result_df['年化波动率'].map(lambda x: format(x, '.2%'))
    # 整理完成的单基金多个时间区间风险收益指标的标准化呈现结果
result_df = result_df[[ '绝对收益率', '年化收益率', '区间最大回撤',
                            '年化波动率', '年化夏普', '卡玛比率', '下行波动率','胜率']]
if st.checkbox('展示指标计算结果'):
    st.subheader('指标计算结果')
    st.dataframe(result_df)

# 画净值图用数据,全部样本区间的数据,包括起始日期和结束日期
if st.checkbox('展示图表'):
    if len(nav_df_part['accum_nav'])>2:
        figdata1 = nav_df_part
        figdata=figdata1.reset_index()
        figdata['time']=pd.to_datetime(figdata['time'])

        fig = go.Figure()
        # 计算动态回测曲线
        figdata['maxdown_curve_navadj'] = -((figdata['accum_nav'].cummax() - figdata['accum_nav']) /
                                            (figdata['accum_nav'].cummax()))
        # 添加产品净值曲线
        fig.add_trace(go.Scatter(
            x=figdata['time'],
            y=figdata['nav_unit'],
            marker=dict(color='#9099ff'),
            name=code,
            xaxis='x3',
            yaxis='y3'))

        # 添加指数走势曲线
        fig.add_trace(go.Scatter(
            x=figdata['time'],
            y=figdata['close_unit'],
            marker=dict(color='#a099af'),
            name=index,
            xaxis='x3',
            yaxis='y3'))

        # 添加净值动态回撤曲线
        fig.add_trace(go.Scatter(
            x=figdata['time'],
            y=figdata['maxdown_curve_navadj'],
            fill='tozeroy',
            name='累计单位净值回撤',
            xaxis='x2',
            yaxis='y2'))
        fig.update_layout(
            title_text=code + "业绩走势及风险收益表现",
            height=1300,
            margin=dict(l=100, r=100, t=60, b=80),
            yaxis={'domain': [0, 0.3]},
            xaxis3={'anchor': "y3", 'tickangle': -70, 'rangeslider': dict(
                visible=True,
                bgcolor='#c5c5c5',
                bordercolor='#888888',  # 边框颜色
                borderwidth=1,
                thickness=0.03,  # 边的宽度
            )},
            legend=dict(
                bgcolor="LightSteelBlue",  # 图例背景颜色
                yanchor="auto",
                y=0.99,  # 设置图例的y轴位置
                xanchor="left",
                x=0.01),  # 图例x轴位置
            yaxis3={'domain': [0.7, 1], 'anchor': 'y3',
                    'title': '净值(归1)', 'tickformat': '.2f'},
            xaxis2={'anchor': "y2", 'tickangle': -70},
            yaxis2={'domain': [0.38, 0.58], 'anchor': 'y2', 'title': '动态回撤', 'tickformat': '.1%'})
        st.plotly_chart(fig)
