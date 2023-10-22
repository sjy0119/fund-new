import streamlit as st
import pandas as pd
import akshare as ak
from akshare.utils import demjson
import plotly.graph_objects as go
import asyncio
import aiohttp

@st.cache_data
def load_data():
   df=pd.read_csv("基金经理")
   return df
df=load_data()

manager_name=df['基金经理名称']

option = st.selectbox(
    '请选择基金经理的姓名',
    tuple(manager_name))

id=df.loc[df['基金经理名称']==option]['基金经理id'].values[0]

def ld_data():
    data_info=pd.read_html(f'http://fund.eastmoney.com/manager/{id}.html')
    return data_info

data_info=ld_data()

fund_all=data_info[1]
fund_all['基金代码']=fund_all['基金代码'].apply(lambda x: '00000'+str(x)).apply(lambda x: x[-6:])
fund_pre=data_info[-1]
fund_pre['基金代码']=fund_pre['基金代码'].apply(lambda x: '00000'+str(x)).apply(lambda x: x[-6:])
fund_all['任职时间']=fund_all['任职时间'].apply(lambda x: str(x))

fund_id=list(fund_all['基金代码'])



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
                #temp_df=temp_df.set_index('净值日期')
                dat.append(temp_df)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = [async_get_url(fund) for fund in fund_id]
    loop.run_until_complete(asyncio.wait(tasks))
    return dat

@st.cache_data(ttl=600)
def get_data(fund_id):
    df=get(fund_id)
    return df

dp=get_data(fund_id)

dp1=[stk_fd_dt.set_index("净值日期") for stk_fd_dt in dp if len(stk_fd_dt)!=0]


fund_net=pd.concat(dp1,axis=1).reset_index('净值日期').sort_values(by='净值日期')
fund_id1=fund_net.columns[1:]
fund_time=[fund_all.loc[fund_all['基金代码']==i]['任职时间'].values[0][:10] for i in fund_id1]
end_time=[fund_all.loc[fund_all['基金代码']==i]['任职时间'].values[0][13:] for i in fund_id1]
fund_net['净值日期']=pd.to_datetime(fund_net['净值日期'])
fund_net=fund_net.set_index('净值日期')

all=[]
for i,j,z in zip(fund_net.columns,fund_time,end_time):
    tr=fund_net[[i]]
    if len(z)>3:
        tr=tr[j:z]
    else:
        tr=tr[j:]
    all.append(tr)

fund_al=pd.concat(all,axis=1).reset_index('净值日期').sort_values(by='净值日期')
for i in fund_al.columns[1:]:
    fund_al[i+'pct']=fund_al[i].pct_change()
    fund_al[i+'cumulative']=(1+fund_al[i+'pct']).cumprod()-1

line = [go.Scatter(x=fund_al['净值日期'],
                                     y=fund_al[i+'cumulative'], mode='lines', name=i) for i in fund_id1]



if option:
    info=df.loc[df['基金经理名称']==option]['基金经理简介'].values[0]
    info1=info.split('。')

    mag=df.loc[df['基金经理名称']==option]['照片链接'].values[0]


    col1, col2= st.columns(2)

    with col1:
        st.header(option)
        st.image(mag)

    with col2:
        st.header("基金经理基本信息")
        st.write(info1)
    with st.container():

        st.header(f"{option}管理基金一览")

        
        st.dataframe(fund_all,hide_index=True)

    with st.container():

        st.header(f"{option}现任基金业绩与排名详情")

        
        st.dataframe(fund_pre,hide_index=True)

    fig_nav_CIS = go.Figure(data=line)

    fig_nav_CIS .update_layout(
                title_text="自担任以来的所管基金累计收益走势 <br> 最新净值日期:" +
                str(fund_al['净值日期'].iloc[-1]),
                margin=dict(l=100, r=100, t=60, b=80),
                yaxis={'tickformat': '.2f', 'title': '收益'},
                xaxis={'tickangle': -70, 'tickformat': '%Y-%m-%d'})
    st.plotly_chart(fig_nav_CIS)

