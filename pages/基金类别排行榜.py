import pandas as pd
import datetime
from akshare.utils import demjson
import asyncio
import aiohttp
import streamlit as st
from plotly.figure_factory import create_table



pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

st.set_page_config(page_icon="😎",)
st.markdown("# 基金类别明细")
st.sidebar.header("基金类别明细")

name_str = st.selectbox(
    '请选择基金类别',
    ("沪深指数-被动指数","沪深指数-增强型","行业主题-被动指数","行业主题-增强型",

           "大盘指数-被动指数","大盘指数-增强型","中小盘指数-被动指数",'中小盘指数-增强型',

           "债券指数-被动指数",'债券指数-增强型',"长期纯债-杠杆0-100",'长期纯债-杠杆100-150',

           "长期纯债-杠杆150-200","短期纯债-杠杆0-100","短期纯债-杠杆100-150","混合债基-杠杆0-100","混合债基-杠杆100-150",

           "定期开放债券-杠杆0-100","定期开放债券-杠杆100-150",'定期开放债券-杠杆150-200','可转债-杠杆0-100',

           '可转债-杠杆100-150','QDII-全球股票','QDII-亚太股票','QDII-大中华区股票',"QDII-美国股票",

           "QDII-债券","QDII-商品","混合型","股票型"
    ))

current_date = datetime.datetime.now().date().isoformat()
last_date = str(int(current_date[:4]) - 1) + current_date[4:]

url_list=[f'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=zs&rs=&gs=0&sc=1nzf&st=desc&sd={last_date}&ed={current_date}&qdii=053|051&tabSubtype=041,,053,051,,&pi=1&pn=100&dx=1&v=0.5612189555404814',
          f'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=zs&rs=&gs=0&sc=1nzf&st=desc&sd={last_date}&ed={current_date}&qdii=053|052&tabSubtype=041,,053,052,,&pi=1&pn=100&dx=1&v=0.9627762584855308',
          f'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=zs&rs=&gs=0&sc=1nzf&st=desc&sd={last_date}&ed={current_date}&qdii=054|051&tabSubtype=041,,054,051,,&pi=1&pn=200&dx=1&v=0.4066042638797256',
          f'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=zs&rs=&gs=0&sc=1nzf&st=desc&sd={last_date}&ed={current_date}&qdii=054|052&tabSubtype=041,,054,052,,&pi=1&pn=200&dx=1&v=0.545713030329777',
          f'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=zs&rs=&gs=0&sc=1nzf&st=desc&sd={last_date}&ed={current_date}&qdii=01|051&tabSubtype=041,,01,051,,&pi=1&pn=200&dx=1&v=0.13852119762588933',
          f'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=zs&rs=&gs=0&sc=1nzf&st=desc&sd={last_date}&ed={current_date}&qdii=01|052&tabSubtype=041,,01,052,,&pi=1&pn=200&dx=1&v=0.45752940267917297',
          f'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=zs&rs=&gs=0&sc=1nzf&st=desc&sd={last_date}&ed={current_date}&qdii=02,03|051&tabSubtype=041,,02,03,051,,&pi=1&pn=200&dx=1&v=0.7043360354032364',
          f'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=zs&rs=&gs=0&sc=1nzf&st=desc&sd={last_date}&ed={current_date}&qdii=02,03|052&tabSubtype=041,,02,03,052,,&pi=1&pn=200&dx=1&v=0.4540073381369334',
          f'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=zs&rs=&gs=0&sc=1nzf&st=desc&sd={last_date}&ed={current_date}&qdii=003|051&tabSubtype=041,,003,051,,&pi=1&pn=200&dx=1&v=0.12445426932998527',
          f'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=zs&rs=&gs=0&sc=1nzf&st=desc&sd={last_date}&ed={current_date}&qdii=003|052&tabSubtype=041,,003,052,,&pi=1&pn=50&dx=1&v=0.9222341759314145',
          f'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=zq&rs=&gs=0&sc=1nzf&st=desc&sd={last_date}&ed={current_date}&qdii=041|0-100&tabSubtype=041,0-100,003,052,,&pi=1&pn=100&dx=1&v=0.9273572580605014',
          f'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=zq&rs=&gs=0&sc=1nzf&st=desc&sd={last_date}&ed={current_date}&qdii=041|100-150&tabSubtype=041,100-150,003,052,,&pi=1&pn=100&dx=1&v=0.36626783862528733',
          f'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=zq&rs=&gs=0&sc=1nzf&st=desc&sd={last_date}&ed={current_date}&qdii=041|150-200&tabSubtype=041,150-200,003,052,,&pi=1&pn=50&dx=1&v=0.5273004472445675',
          f'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=zq&rs=&gs=0&sc=1nzf&st=desc&sd={last_date}&ed={current_date}&qdii=042|0-100&tabSubtype=042,0-100,003,052,,&pi=1&pn=100&dx=1&v=0.7754633479762327',
          f'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=zq&rs=&gs=0&sc=1nzf&st=desc&sd={last_date}&ed={current_date}&qdii=042|100-150&tabSubtype=042,100-150,003,052,,&pi=1&pn=100&dx=1&v=0.4358363554061009',
          f'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=zq&rs=&gs=0&sc=1nzf&st=desc&sd={last_date}&ed={current_date}&qdii=043|0-100&tabSubtype=043,0-100,003,052,,&pi=1&pn=100&dx=1&v=0.8043668828045416',
          f'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=zq&rs=&gs=0&sc=1nzf&st=desc&sd={last_date}&ed={current_date}&qdii=043|100-150&tabSubtype=043,100-150,003,052,,&pi=1&pn=100&dx=1&v=0.6998457475896478',
          f'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=zq&rs=&gs=0&sc=1nzf&st=desc&sd={last_date}&ed={current_date}&qdii=008|0-100&tabSubtype=008,0-100,003,052,,&pi=1&pn=100&dx=1&v=0.1045904586445372',
          f'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=zq&rs=&gs=0&sc=1nzf&st=desc&sd={last_date}&ed={current_date}&qdii=008|100-150&tabSubtype=008,100-150,003,052,,&pi=1&pn=100&dx=1&v=0.4750601624587343',
          f'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=zq&rs=&gs=0&sc=1nzf&st=desc&sd={last_date}&ed={current_date}&qdii=008|150-200&tabSubtype=008,150-200,003,052,,&pi=1&pn=50&dx=1&v=0.9158595865229342',
          f'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=zq&rs=&gs=0&sc=1nzf&st=desc&sd={last_date}&ed={current_date}&qdii=045|0-100&tabSubtype=045,0-100,003,052,,&pi=1&pn=100&dx=1&v=0.3957957120238935',
          f'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=zq&rs=&gs=0&sc=1nzf&st=desc&sd={last_date}&ed={current_date}&qdii=045|100-150&tabSubtype=045,100-150,003,052,,&pi=1&pn=100&dx=1&v=0.5451794710903861',
          f'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=qdii&rs=&gs=0&sc=1nzf&st=desc&sd={last_date}&ed={current_date}&qdii=311&tabSubtype=045,150-200,003,052,,&pi=1&pn=100&dx=1&v=0.5404930594605',
          f'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=qdii&rs=&gs=0&sc=1nzf&st=desc&sd={last_date}&ed={current_date}&qdii=312&tabSubtype=045,150-200,003,052,,&pi=1&pn=100&dx=1&v=0.3149835927662821',
          f'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=qdii&rs=&gs=0&sc=1nzf&st=desc&sd={last_date}&ed={current_date}&qdii=313&tabSubtype=045,150-200,003,052,,&pi=1&pn=100&dx=1&v=0.2314686194654152',
          f'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=qdii&rs=&gs=0&sc=1nzf&st=desc&sd={last_date}&ed={current_date}&qdii=317&tabSubtype=045,150-200,003,052,,&pi=1&pn=100&dx=1&v=0.287568220736645',
          f'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=qdii&rs=&gs=0&sc=1nzf&st=desc&sd={last_date}&ed={current_date}&qdii=330&tabSubtype=045,150-200,003,052,,&pi=1&pn=100&dx=1&v=0.3447163326985554',
          f'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=qdii&rs=&gs=0&sc=1nzf&st=desc&sd={last_date}&ed={current_date}&qdii=340&tabSubtype=045,150-200,003,052,,&pi=1&pn=100&dx=1&v=0.9427080941011512',
          f'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=hh&rs=&gs=0&sc=1nzf&st=desc&sd={last_date}&ed={current_date}&qdii=340&tabSubtype=045,150-200,003,052,,&pi=1&pn=300&dx=1&v=0.39672188146966714',
          f'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=gp&rs=&gs=0&sc=1nzf&st=desc&sd={last_date}&ed={current_date}&qdii=340&tabSubtype=045,150-200,003,052,,&pi=1&pn=300&dx=1&v=0.8855444003009028'

]

name_list=["沪深指数-被动指数","沪深指数-增强型","行业主题-被动指数","行业主题-增强型",

           "大盘指数-被动指数","大盘指数-增强型","中小盘指数-被动指数",'中小盘指数-增强型',

           "债券指数-被动指数",'债券指数-增强型',"长期纯债-杠杆0-100",'长期纯债-杠杆100-150',

           "长期纯债-杠杆150-200","短期纯债-杠杆0-100","短期纯债-杠杆100-150","混合债基-杠杆0-100","混合债基-杠杆100-150",

           "定期开放债券-杠杆0-100","定期开放债券-杠杆100-150",'定期开放债券-杠杆150-200','可转债-杠杆0-100',

           '可转债-杠杆100-150','QDII-全球股票','QDII-亚太股票','QDII-大中华区股票',"QDII-美国股票",

           "QDII-债券","QDII-商品","混合型","股票型"
]
def get(url_list,name_list):
    dat=dict()
    async def async_get_url(url,i):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36",
            "Referer": "http://fund.eastmoney.com/fundguzhi.html",
        }
        async with aiohttp.ClientSession() as session:  # 解释1
            async with session.get(url, headers=headers) as r:
                text_data = await r.text()
                json_data = demjson.decode(text_data[text_data.find("{") : -1])
                temp_df = pd.DataFrame(json_data["datas"])
                temp_df = temp_df.iloc[:, 0].str.split(",",expand=True)
                temp_df.reset_index(inplace=True)
                temp_df["index"] = list(range(1, len(temp_df) + 1))
                temp_df.columns = [
                    "序号",
                    "基金代码",
                    "基金简称",
                    "_",
                    "日期",
                    "单位净值",
                    "累计净值",
                    "日增长率",
                    "近1周",
                    "近1月",
                    "近3月",
                    "近6月",
                    "近1年",
                    "近2年",
                    "近3年",
                    "今年来",
                    "成立来",
                    "_",
                    "_",
                    "自定义",
                    "_",
                    "手续费",
                    "_",
                    "_",
                    "_",
                    "_",
                ]
                temp_df = temp_df[
                    [
                        "序号",
                        "基金代码",
                        "基金简称",
                        "日期",
                        "单位净值",
                        "累计净值",
                        "日增长率",
                        "近1周",
                        "近1月",
                        "近3月",
                        "近6月",
                        "近1年",
                        "近2年",
                        "近3年",
                        "今年来",
                        "成立来",
                        "手续费",
                    ]
                ]
                temp_df=temp_df.loc[temp_df['近3年']!='']
                temp_df=temp_df.apply(pd.to_numeric,errors='ignore')
                temp_df['基金代码']=temp_df['基金代码'].apply(lambda x:"0000"+str(x))
                temp_df['基金代码']=temp_df['基金代码'].apply(lambda x: x[-6:])
                dat[i]=temp_df
             
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = [async_get_url(url,i) for url,i in zip(url_list,name_list)]
    loop.run_until_complete(asyncio.wait(tasks))
    return dat

@st.cache_data(ttl=660)
def get_data(url_list,name_list):
    df=get(url_list,name_list)
    return df

df=get_data(url_list,name_list)


if name_str=="沪深指数-被动指数":
    st.dataframe(df["沪深指数-被动指数"])
    st.caption('可查看此类基金的跟踪指数及表现情况')
    method = st.selectbox(
    '请选择基金',
    (i for i in df["沪深指数-被动指数"]['基金简称']))
    if method:
        n_=df["沪深指数-被动指数"]
        n_1=n_.loc[n_['基金简称']==method]['基金代码'].values[0]
        rtf=pd.read_html(f'http://fundf10.eastmoney.com/tsdata_{n_1}.html')

        table1=create_table(rtf[1])
        table2=create_table(rtf[2])

        st.plotly_chart(table1)
        st.plotly_chart(table2)

elif name_str=="沪深指数-增强型":
    st.dataframe(df["沪深指数-增强型"])
    st.caption('可查看此类基金的跟踪指数及表现情况')
    method = st.selectbox(
    '请选择基金',
    (i for i in df["沪深指数-增强型"]['基金简称']))
    if method:
        n_=df["沪深指数-增强型"]
        n_1=n_.loc[n_['基金简称']==method]['基金代码'].values[0]
        rtf=pd.read_html(f'http://fundf10.eastmoney.com/tsdata_{n_1}.html')

        table1=create_table(rtf[1])
        table2=create_table(rtf[2])

        st.plotly_chart(table1)
        st.plotly_chart(table2)
elif name_str=="行业主题-被动指数":
    st.dataframe(df["行业主题-被动指数"])
    st.caption('可查看此类基金的跟踪指数及表现情况')
    method = st.selectbox(
    '请选择基金',
    (i for i in df["行业主题-被动指数"]['基金简称']))
    if method:
        n_=df["行业主题-被动指数"]
        n_1=n_.loc[n_['基金简称']==method]['基金代码'].values[0]
        rtf=pd.read_html(f'http://fundf10.eastmoney.com/tsdata_{n_1}.html')

        table1=create_table(rtf[1])
        table2=create_table(rtf[2])

        st.plotly_chart(table1)
        st.plotly_chart(table2)
elif name_str=="行业主题-增强型":
    st.dataframe(df["行业主题-增强型"])
    st.caption('可查看此类基金的跟踪指数及表现情况')
    method = st.selectbox(
    '请选择基金',
    (i for i in df["行业主题-增强型"]['基金简称']))
    if method:
        n_=df["行业主题-增强型"]
        n_1=n_.loc[n_['基金简称']==method]['基金代码'].values[0]
        rtf=pd.read_html(f'http://fundf10.eastmoney.com/tsdata_{n_1}.html')

        table1=create_table(rtf[1])
        table2=create_table(rtf[2])

        st.plotly_chart(table1)
        st.plotly_chart(table2)


elif name_str== "大盘指数-被动指数":
    st.dataframe(df[ "大盘指数-被动指数"])
    st.caption('可查看此类基金的跟踪指数及表现情况')
    method = st.selectbox(
    '请选择基金',
    (i for i in df[ "大盘指数-被动指数"]['基金简称']))
    if method:
        n_=df[ "大盘指数-被动指数"]
        n_1=n_.loc[n_['基金简称']==method]['基金代码'].values[0]
        rtf=pd.read_html(f'http://fundf10.eastmoney.com/tsdata_{n_1}.html')

        table1=create_table(rtf[1])
        table2=create_table(rtf[2])

        st.plotly_chart(table1)
        st.plotly_chart(table2)
elif name_str=="大盘指数-增强型":
    st.dataframe(df["大盘指数-增强型"])
    st.caption('可查看此类基金的跟踪指数及表现情况')
    method = st.selectbox(
    '请选择基金',
    (i for i in df["大盘指数-增强型"]['基金简称']))
    if method:
        n_=df["大盘指数-增强型"]
        n_1=n_.loc[n_['基金简称']==method]['基金代码'].values[0]
        rtf=pd.read_html(f'http://fundf10.eastmoney.com/tsdata_{n_1}.html')

        table1=create_table(rtf[1])
        table2=create_table(rtf[2])

        st.plotly_chart(table1)
        st.plotly_chart(table2)
elif name_str=="中小盘指数-被动指数":
    st.dataframe(df["中小盘指数-被动指数"])
    st.caption('可查看此类基金的跟踪指数及表现情况')
    method = st.selectbox(
    '请选择基金',
    (i for i in df["中小盘指数-被动指数"]['基金简称']))
    if method:
        n_=df["中小盘指数-被动指数"]
        n_1=n_.loc[n_['基金简称']==method]['基金代码'].values[0]
        rtf=pd.read_html(f'http://fundf10.eastmoney.com/tsdata_{n_1}.html')

        table1=create_table(rtf[1])
        table2=create_table(rtf[2])

        st.plotly_chart(table1)
        st.plotly_chart(table2)
elif name_str=='中小盘指数-增强型':
    st.dataframe(df['中小盘指数-增强型'])
    st.caption('可查看此类基金的跟踪指数及表现情况')
    method = st.selectbox(
    '请选择基金',
    (i for i in df['中小盘指数-增强型']['基金简称']))
    if method:
        n_=df['中小盘指数-增强型']
        n_1=n_.loc[n_['基金简称']==method]['基金代码'].values[0]
        rtf=pd.read_html(f'http://fundf10.eastmoney.com/tsdata_{n_1}.html')

        table1=create_table(rtf[1])
        table2=create_table(rtf[2])

        st.plotly_chart(table1)
        st.plotly_chart(table2)


elif name_str=="债券指数-被动指数":
    st.dataframe(df["债券指数-被动指数"])
    st.caption('可查看此类基金的跟踪指数及表现情况')
    method = st.selectbox(
    '请选择基金',
    (i for i in df["债券指数-被动指数"]['基金简称']))
    if method:
        n_=df["债券指数-被动指数"]
        n_1=n_.loc[n_['基金简称']==method]['基金代码'].values[0]
        rtf=pd.read_html(f'http://fundf10.eastmoney.com/tsdata_{n_1}.html')

        table1=create_table(rtf[1])
        table2=create_table(rtf[2])

        st.plotly_chart(table1)
        st.plotly_chart(table2)
elif name_str=='债券指数-增强型':
    st.dataframe(df['债券指数-增强型'])
    st.caption('可查看此类基金的跟踪指数及表现情况')
    method = st.selectbox(
    '请选择基金',
    (i for i in df['债券指数-增强型']['基金简称']))
    if method:
        n_=df['债券指数-增强型']
        n_1=n_.loc[n_['基金简称']==method]['基金代码'].values[0]
        rtf=pd.read_html(f'http://fundf10.eastmoney.com/tsdata_{n_1}.html')

        table1=create_table(rtf[1])
        table2=create_table(rtf[2])

        st.plotly_chart(table1)
        st.plotly_chart(table2)

elif name_str=="长期纯债-杠杆0-100":
    st.dataframe(df["长期纯债-杠杆0-100"])
elif name_str=='长期纯债-杠杆100-150':
    st.dataframe(df['长期纯债-杠杆100-150'])
elif name_str=="长期纯债-杠杆150-200":
    st.dataframe(df["长期纯债-杠杆150-200"])
elif name_str=="短期纯债-杠杆0-100":
    st.dataframe(df["短期纯债-杠杆0-100"])
elif name_str=="短期纯债-杠杆100-150":
    st.dataframe(df["短期纯债-杠杆100-150"])
elif name_str=="混合债基-杠杆0-100":
    st.dataframe(df["混合债基-杠杆0-100"])
elif name_str=="混合债基-杠杆100-150":
    st.dataframe(df["混合债基-杠杆100-150"])
elif name_str=="定期开放债券-杠杆0-100":
    st.dataframe(df["定期开放债券-杠杆0-100"])
elif name_str=="定期开放债券-杠杆100-150":
    st.dataframe(df["定期开放债券-杠杆100-150"])
elif name_str=='定期开放债券-杠杆150-200':
    st.dataframe(df['定期开放债券-杠杆150-200'])
elif name_str=='可转债-杠杆0-100':
    st.dataframe(df['可转债-杠杆0-100'])
elif name_str=='可转债-杠杆100-150':
    st.dataframe(df['可转债-杠杆100-150'])
elif name_str=='QDII-全球股票':
    st.dataframe(df['QDII-全球股票'])
elif name_str=='QDII-亚太股票':
    st.dataframe(df['QDII-亚太股票'])
elif name_str=='QDII-大中华区股票':
    st.dataframe(df['QDII-大中华区股票'])
elif name_str=="QDII-美国股票":
    st.dataframe(df["QDII-美国股票"])
elif name_str=="QDII-债券":
    st.dataframe(df["QDII-债券"])
elif name_str=="QDII-商品":
    st.dataframe(df["QDII-商品"])
elif name_str=='混合型':
    st.dataframe(df['混合型'])
elif name_str=='股票型':
    st.dataframe(df['股票型'])
