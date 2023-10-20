import pandas as pd
import datetime
from akshare.utils import demjson
import asyncio
import aiohttp
import streamlit as st



pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

st.set_page_config(page_icon="😎",)
st.markdown("# 基金类别明细")
st.sidebar.header("基金类别明细")

multi = '''可供输入的选项：股票型，混合型，债券型，指数型，QDII,LOF,FOF'''
st.markdown(multi)

name_str=st.text_input('请输入基金类别名称')

current_date = datetime.datetime.now().date().isoformat()
last_date = str(int(current_date[:4]) - 1) + current_date[4:]

type_map = {
        "股票型": ["gp", "6yzf"],
        "混合型": ["hh", "6yzf"],
        "债券型": ["zq", "6yzf"],
        "指数型": ["zs", "6yzf"],
        "QDII": ["qdii", "6yzf"],
        "LOF": ["lof", "6yzf"],
        "FOF": ["fof", "6yzf"]}

type_list=["gp","hh","zq","zs","qdii","lof","fof"]

params_list=[{
    "op": "ph",
    "dt": "kf",
    "ft": ip,
    "rs": "",
    "gs": "0",
    "sc": "6yzf",
    "st": "desc",
    "sd": last_date,
    "ed": current_date,
    "qdii": "",
    "tabSubtype": ",,,,,",
    "pi": "1",
    "pn": "20000",
    "dx": "1",
    "v": "0.1591891419018292",
} for ip in type_list]


def get():
    dat=dict()
    async def async_get_url(params,i):
        url="http://fund.eastmoney.com/data/rankhandler.aspx"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36",
            "Referer": "http://fund.eastmoney.com/fundguzhi.html",
        }
        async with aiohttp.ClientSession() as session:  # 解释1
            async with session.get(url, params=params,headers=headers) as r:
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
                    #print('获取成功')
            #nest_asyncio.apply()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = [async_get_url(params,i) for params,i in zip(params_list,type_list)]
    loop.run_until_complete(asyncio.wait(tasks))
    return dat

@st.cache_data
def get_data():
    df=get()
    return df

df=get_data()

if name_str=='股票型':
    st.dataframe(df['gp'])
elif name_str=='混合型':
    st.dataframe(df['hh'])
elif name_str=='债券型':
    st.dataframe(df['zq'])
elif name_str=='指数型':
    st.dataframe(df['zs'])
elif name_str=='QDII':
    st.dataframe(df['qdii'])
elif name_str=='LOF':
    st.dataframe(df['lof'])
elif name_str=='FOF':
    st.dataframe(df['fof'])
