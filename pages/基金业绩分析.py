import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import akshare as ak
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] =False 


st.set_page_config(page_icon="😎",)
st.markdown("# 基金业绩分析")
st.sidebar.header("基金业绩分析")

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
year=str(date.today().year)
list1=[-5,-22,-66,-132]

def nav_analy(nav_df_part):
    adj_nav_end = list(nav_df_part[code])[-1]  # 复权累计净值的区间末位数值
    adj_nav_start = list(nav_df_part[code])[0]  # 复权累计净值的区间首位数
    # 样本期的绝对收益率
    abs_ret = (adj_nav_end/adj_nav_start)-1
        #样本期的年化收益率
    annual_ret = pow(adj_nav_end/adj_nav_start, 250/(len(nav_df_part)-1))-1

    #计算胜率
    fenmu=len(nav_df_part)
    sd=nav_df_part.loc[nav_df_part[code+'pct']>0]
    fenzi=len(sd)
    victory_days=fenzi/fenmu
    #样本期的最大回撤
    #nav_df_part=nav_one
    interval_max_down = ((nav_df_part[code].cummax()-nav_df_part[code]) /
                        (nav_df_part[code].cummax())).max()

    # 样本期年化波动率

    annual_var = nav_df_part[code+'pct'].std(
            ddof=1)*pow(250, 0.5)

    # 样本期间年化夏普，年化后的平均收益率-无风险利率 /年化后的波动率
    rf_rate=0.02
    annual_sharpe = (
            pow((1+nav_df_part[code+'pct'].mean()), 250)-1-rf_rate)/annual_var
    #计算卡玛比率
    interval_calmar = annual_ret/interval_max_down

    # 样本期下行波动率
    temp = nav_df_part[nav_df_part[code+'pct']
                        < nav_df_part[code+'pct'].mean()]
    temp2 = temp[code+'pct'] - \
            nav_df_part[code+'pct'].mean()
    down_var = np.sqrt((temp2**2).sum()/(len(nav_df_part)-1))*pow(250, 0.5)

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
    result_df['下行波动率'] = result_df['下行波动率'].map(lambda x: format(x, '.2%'))
    result_df['胜率'] = result_df['胜率'].map(lambda x: format(x, '.2%'))
    result_df = result_df[[ '绝对收益率', '年化收益率', '区间最大回撤',
                            '年化波动率', '年化夏普', '卡玛比率', '下行波动率','胜率']]
    return result_df

@st.cache_data
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
@st.cache_data
def load_data1(code):
    fund_df = ak.fund_open_fund_info_em(fund=code, indicator="同类排名走势").rename(columns={'报告日期':'date'})
    return fund_df



if code:
    df=load_data(code,index)

    rank=load_data1(code)

    list1=[-5,-22,-66,-132]
    group_list=[df.iloc[list1[i]:,:] for i in range(4)]+[df[year:year],df]

    year_return=pd.DataFrame()
    list_year=[str(i) for i in df.index.year.unique()]
    fund=[list(df[i:i][code])[-1]/list(df[i:i][code])[0]-1 for i in list_year]
    basic=[list(df[i:i][index])[-1]/list(df[i:i][index])[0]-1 for i in list_year]
    year_return['年份']=list_year
    year_return['基金收益']=fund
    year_return['基准收益']=basic
    year_return['基金收益'] = year_return['基金收益'].map(lambda x: format(x, '.2%'))
    year_return['基准收益'] = year_return['基准收益'].map(lambda x: format(x, '.2%'))

    data_list=[nav_analy(df) for df in group_list]
    data=pd.concat(data_list)
    data.insert(0,'阶段',['近一周','近一个月','近三个月','近六个月','今年以来','成立以来'])

# 画净值图用数据,全部样本区间的数据,包括起始日期和结束日期
    figdata1=df[开始:结束]
    figdata1['nav_unit']=figdata1[code]/figdata1[code][0]
    figdata1['close_unit']=figdata1[index]/figdata1[index][0]
    
    if len(df[code])>2:
        figdata=figdata1.reset_index()
        figdata=figdata.rename(columns={'date':'time'})
        figdata['time']=pd.to_datetime(figdata['time'])

        figdata['maxdown_curve_navadj'] = -((figdata[code].cummax() - figdata[code]) /
                                        (figdata[code].cummax()))
        line0 = go.Scatter(x=figdata['time'],y=figdata['nav_unit'], mode='lines', name=code)
        line1 = go.Scatter(x=figdata['time'],y=figdata['close_unit'], mode='lines', name=index)
        fig_nav_CIS = go.Figure(data=[line0,line1])

        fig_nav_CIS .update_layout(
                    title_text="基金净值与基准净值走势 <br>(归一化处理)",
                    yaxis={'tickformat': '.2f', 'title': ' 净值'},
                    xaxis={'tickangle': -70, 'tickformat': '%Y-%m-%d'})
        
        fig1 = go.Figure(data=[
                go.Bar(name=code, x=list(year_return['年份']), y=list(year_return['基金收益'])),
                go.Bar(name=index, x=list(year_return['年份']), y=list(year_return['基准收益']))
                ])

                # 柱状图模式需要设置：4选1
        fig1.update_layout(barmode='group',title_text='年度基金与基准涨幅')  # ['stack', 'group', 'overlay', 'relative']

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=figdata['time'],
            y=figdata['maxdown_curve_navadj'],
            fill='tozeroy',
            name='累计单位净值动态回撤',
            xaxis='x2',
            yaxis='y2'))
        fig2.update_layout(
            title_text=code + "回撤情况",
            )
        
        line2 = go.Scatter(x=rank['date'],y=rank['同类型排名-每日近三月排名'], mode='lines', name='同类型排名-每日近三月排名')
        line3 = go.Scatter(x=rank['date'],y=rank['总排名-每日近三月排名'], mode='lines', name='总排名-每日近三月排名')
        fig_ = go.Figure(data=[line2,line3])

        fig_ .update_layout(
                    title_text="基金排名走势 ",
                    yaxis={'tickformat': '.2f', 'title': '排名'},
                    xaxis={'tickangle': -70, 'tickformat': '%Y-%m-%d'})

        
        st.plotly_chart(fig_nav_CIS)

        st.plotly_chart(fig2)

        st.dataframe(data,hide_index=True)

        st.plotly_chart(fig1)

        st.plotly_chart(fig_)
