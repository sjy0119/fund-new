import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
from plotly.figure_factory import create_table
from datetime import datetime
import requests
import orjson
import tushare as ts
pro = ts.pro_api('8e812052c92d7a829f0e3b0197d248e48bb2ba3efbbaa60f505e6852')
import warnings
warnings.filterwarnings("ignore")
from py_mini_racer import py_mini_racer

from akshare.stock_feature.stock_a_indicator import get_cookie_csrf

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] =False 

st.title('市场看板 :blue[!] :sunglasses:')
st.markdown('1.市场指数情况')
st.markdown('2.股票市场情况')
st.markdown('3.风格因子情况')
st.markdown('4.宏观指标情况及私募策略指数情况')

#爬虫函数定义模块
hash_code = """
function e(n) {
    var e, t, r = "", o = -1, f;
    if (n && n.length) {
        f = n.length;
        while ((o += 1) < f) {
            e = n.charCodeAt(o);
            t = o + 1 < f ? n.charCodeAt(o + 1) : 0;
            if (55296 <= e && e <= 56319 && 56320 <= t && t <= 57343) {
                e = 65536 + ((e & 1023) << 10) + (t & 1023);
                o += 1
            }
            if (e <= 127) {
                r += String.fromCharCode(e)
            } else if (e <= 2047) {
                r += String.fromCharCode(192 | e >>> 6 & 31, 128 | e & 63)
            } else if (e <= 65535) {
                r += String.fromCharCode(224 | e >>> 12 & 15, 128 | e >>> 6 & 63, 128 | e & 63)
            } else if (e <= 2097151) {
                r += String.fromCharCode(240 | e >>> 18 & 7, 128 | e >>> 12 & 63, 128 | e >>> 6 & 63, 128 | e & 63)
            }
        }
    }
    return r
}

function t(n) {
    var e, t, r, o, f, i = [], h;
    e = t = r = o = f = 0;
    if (n && n.length) {
        h = n.length;
        n += "";
        while (e < h) {
            r = n.charCodeAt(e);
            t += 1;
            if (r < 128) {
                i[t] = String.fromCharCode(r);
                e += 1
            } else if (r > 191 && r < 224) {
                o = n.charCodeAt(e + 1);
                i[t] = String.fromCharCode((r & 31) << 6 | o & 63);
                e += 2
            } else {
                o = n.charCodeAt(e + 1);
                f = n.charCodeAt(e + 2);
                i[t] = String.fromCharCode(
                    (r & 15) << 12 | (o & 63) << 6 | f & 63);
                e += 3
            }
        }
    }
    return i.join("")
}

function r(n, e) {
    var t = (n & 65535) + (e & 65535)
        , r = (n >> 16) + (e >> 16) + (t >> 16);
    return r << 16 | t & 65535
}

function o(n, e) {
    return n << e | n >>> 32 - e
}

function f(n, e) {
    var t = e ? "0123456789ABCDEF" : "0123456789abcdef", r = "", o, f = 0, i = n.length;
    for (; f < i; f += 1) {
        o = n.charCodeAt(f);
        r += t.charAt(o >>> 4 & 15) + t.charAt(o & 15)
    }
    return r
}

function i(n) {
    var e, t = n.length, r = "";
    for (e = 0; e < t; e += 1) {
        r += String.fromCharCode(n.charCodeAt(e) & 255, n.charCodeAt(e) >>> 8 & 255)
    }
    return r
}

function h(n) {
    var e, t = n.length, r = "";
    for (e = 0; e < t; e += 1) {
        r += String.fromCharCode(n.charCodeAt(e) >>> 8 & 255, n.charCodeAt(e) & 255)
    }
    return r
}

function u(n) {
    var e, t = n.length * 32, r = "";
    for (e = 0; e < t; e += 8) {
        r += String.fromCharCode(n[e >> 5] >>> 24 - e % 32 & 255)
    }
    return r
}

function a(n) {
    var e, t = n.length * 32, r = "";
    for (e = 0; e < t; e += 8) {
        r += String.fromCharCode(n[e >> 5] >>> e % 32 & 255)
    }
    return r
}

function c(n) {
    var e, t = n.length * 8, r = Array(n.length >> 2), o = r.length;
    for (e = 0; e < o; e += 1) {
        r[e] = 0
    }
    for (e = 0; e < t; e += 8) {
        r[e >> 5] |= (n.charCodeAt(e / 8) & 255) << e % 32
    }
    return r
}

function l(n) {
    var e, t = n.length * 8, r = Array(n.length >> 2), o = r.length;
    for (e = 0; e < o; e += 1) {
        r[e] = 0
    }
    for (e = 0; e < t; e += 8) {
        r[e >> 5] |= (n.charCodeAt(e / 8) & 255) << 24 - e % 32
    }
    return r
}

function D(n, e) {
    var t = e.length, r = Array(), o, f, i, h, u, a, c, l;
    a = Array(Math.ceil(n.length / 2));
    h = a.length;
    for (o = 0; o < h; o += 1) {
        a[o] = n.charCodeAt(o * 2) << 8 | n.charCodeAt(o * 2 + 1)
    }
    while (a.length > 0) {
        u = Array();
        i = 0;
        for (o = 0; o < a.length; o += 1) {
            i = (i << 16) + a[o];
            f = Math.floor(i / t);
            i -= f * t;
            if (u.length > 0 || f > 0) {
                u[u.length] = f
            }
        }
        r[r.length] = i;
        a = u
    }
    c = "";
    for (o = r.length - 1; o >= 0; o--) {
        c += e.charAt(r[o])
    }
    l = Math.ceil(n.length * 8 / (Math.log(e.length) / Math.log(2)));
    for (o = c.length; o < l; o += 1) {
        c = e[0] + c
    }
    return c
}

function B(n, e) {
    var t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/", r = "", o = n.length, f, i, h;
    e = e || "=";
    for (f = 0; f < o; f += 3) {
        h = n.charCodeAt(f) << 16 | (f + 1 < o ? n.charCodeAt(f + 1) << 8 : 0) | (f + 2 < o ? n.charCodeAt(f + 2) : 0);
        for (i = 0; i < 4; i += 1) {
            if (f * 8 + i * 6 > n.length * 8) {
                r += e
            } else {
                r += t.charAt(h >>> 6 * (3 - i) & 63)
            }
        }
    }
    return r
}

function hex(n) {
    return f(u(n, h), t)
}

function u(n) {
    n = h ? e(n) : n;
    return a(C(c(n), n.length * 8))
}

function l(n, t) {
    var r, o, f, i, u;
    n = h ? e(n) : n;
    t = h ? e(t) : t;
    r = c(n);
    if (r.length > 16) {
        r = C(r, n.length * 8)
    }
    o = Array(16),
        f = Array(16);
    for (u = 0; u < 16; u += 1) {
        o[u] = r[u] ^ 909522486;
        f[u] = r[u] ^ 1549556828
    }
    i = C(o.concat(c(t)), 512 + t.length * 8);
    return a(C(f.concat(i), 512 + 128))
}

function C(n, e) {
    var t, o, f, i, h, u = 1732584193, a = -271733879, c = -1732584194, l = 271733878;
    n[e >> 5] |= 128 << e % 32;
    n[(e + 64 >>> 9 << 4) + 14] = e;
    for (t = 0; t < n.length; t += 16) {
        o = u;
        f = a;
        i = c;
        h = l;
        u = s(u, a, c, l, n[t + 0], 7, -680876936);
        l = s(l, u, a, c, n[t + 1], 12, -389564586);
        c = s(c, l, u, a, n[t + 2], 17, 606105819);
        a = s(a, c, l, u, n[t + 3], 22, -1044525330);
        u = s(u, a, c, l, n[t + 4], 7, -176418897);
        l = s(l, u, a, c, n[t + 5], 12, 1200080426);
        c = s(c, l, u, a, n[t + 6], 17, -1473231341);
        a = s(a, c, l, u, n[t + 7], 22, -45705983);
        u = s(u, a, c, l, n[t + 8], 7, 1770035416);
        l = s(l, u, a, c, n[t + 9], 12, -1958414417);
        c = s(c, l, u, a, n[t + 10], 17, -42063);
        a = s(a, c, l, u, n[t + 11], 22, -1990404162);
        u = s(u, a, c, l, n[t + 12], 7, 1804603682);
        l = s(l, u, a, c, n[t + 13], 12, -40341101);
        c = s(c, l, u, a, n[t + 14], 17, -1502002290);
        a = s(a, c, l, u, n[t + 15], 22, 1236535329);
        u = w(u, a, c, l, n[t + 1], 5, -165796510);
        l = w(l, u, a, c, n[t + 6], 9, -1069501632);
        c = w(c, l, u, a, n[t + 11], 14, 643717713);
        a = w(a, c, l, u, n[t + 0], 20, -373897302);
        u = w(u, a, c, l, n[t + 5], 5, -701558691);
        l = w(l, u, a, c, n[t + 10], 9, 38016083);
        c = w(c, l, u, a, n[t + 15], 14, -660478335);
        a = w(a, c, l, u, n[t + 4], 20, -405537848);
        u = w(u, a, c, l, n[t + 9], 5, 568446438);
        l = w(l, u, a, c, n[t + 14], 9, -1019803690);
        c = w(c, l, u, a, n[t + 3], 14, -187363961);
        a = w(a, c, l, u, n[t + 8], 20, 1163531501);
        u = w(u, a, c, l, n[t + 13], 5, -1444681467);
        l = w(l, u, a, c, n[t + 2], 9, -51403784);
        c = w(c, l, u, a, n[t + 7], 14, 1735328473);
        a = w(a, c, l, u, n[t + 12], 20, -1926607734);
        u = F(u, a, c, l, n[t + 5], 4, -378558);
        l = F(l, u, a, c, n[t + 8], 11, -2022574463);
        c = F(c, l, u, a, n[t + 11], 16, 1839030562);
        a = F(a, c, l, u, n[t + 14], 23, -35309556);
        u = F(u, a, c, l, n[t + 1], 4, -1530992060);
        l = F(l, u, a, c, n[t + 4], 11, 1272893353);
        c = F(c, l, u, a, n[t + 7], 16, -155497632);
        a = F(a, c, l, u, n[t + 10], 23, -1094730640);
        u = F(u, a, c, l, n[t + 13], 4, 681279174);
        l = F(l, u, a, c, n[t + 0], 11, -358537222);
        c = F(c, l, u, a, n[t + 3], 16, -722521979);
        a = F(a, c, l, u, n[t + 6], 23, 76029189);
        u = F(u, a, c, l, n[t + 9], 4, -640364487);
        l = F(l, u, a, c, n[t + 12], 11, -421815835);
        c = F(c, l, u, a, n[t + 15], 16, 530742520);
        a = F(a, c, l, u, n[t + 2], 23, -995338651);
        u = E(u, a, c, l, n[t + 0], 6, -198630844);
        l = E(l, u, a, c, n[t + 7], 10, 1126891415);
        c = E(c, l, u, a, n[t + 14], 15, -1416354905);
        a = E(a, c, l, u, n[t + 5], 21, -57434055);
        u = E(u, a, c, l, n[t + 12], 6, 1700485571);
        l = E(l, u, a, c, n[t + 3], 10, -1894986606);
        c = E(c, l, u, a, n[t + 10], 15, -1051523);
        a = E(a, c, l, u, n[t + 1], 21, -2054922799);
        u = E(u, a, c, l, n[t + 8], 6, 1873313359);
        l = E(l, u, a, c, n[t + 15], 10, -30611744);
        c = E(c, l, u, a, n[t + 6], 15, -1560198380);
        a = E(a, c, l, u, n[t + 13], 21, 1309151649);
        u = E(u, a, c, l, n[t + 4], 6, -145523070);
        l = E(l, u, a, c, n[t + 11], 10, -1120210379);
        c = E(c, l, u, a, n[t + 2], 15, 718787259);
        a = E(a, c, l, u, n[t + 9], 21, -343485551);
        u = r(u, o);
        a = r(a, f);
        c = r(c, i);
        l = r(l, h)
    }
    return Array(u, a, c, l)
}

function A(n, e, t, f, i, h) {
    return r(o(r(r(e, n), r(f, h)), i), t)
}

function s(n, e, t, r, o, f, i) {
    return A(e & t | ~e & r, n, e, o, f, i)
}

function w(n, e, t, r, o, f, i) {
    return A(e & r | t & ~r, n, e, o, f, i)
}

function F(n, e, t, r, o, f, i) {
    return A(e ^ t ^ r, n, e, o, f, i)
}

function E(n, e, t, r, o, f, i) {
    return A(t ^ (e | ~r), n, e, o, f, i)
}
"""
js_functions = py_mini_racer.MiniRacer()
js_functions.eval(hash_code)
token = js_functions.call("hex", datetime.now().date().isoformat()).lower()


def stock_market_pe_lg(symbol: str = "深证") -> pd.DataFrame:
    """
    乐咕乐股-主板市盈率
    https://legulegu.com/stockdata/shanghaiPE
    :param symbol: choice of {"上证", "深证", "创业板", "科创版"}
    :type symbol: str
    :return: 指定市场的市盈率数据
    :rtype: pandas.DataFrame
    """
    if symbol in {"上证", "深证", "创业板"}:
        url = "https://legulegu.com/api/stock-data/market-pe"
        symbol_map = {
            "上证": "1",
            "深证": "2",
            "创业板": "4",
        }
        url_map = {
            "上证": "https://legulegu.com/stockdata/shanghaiPE",
            "深证": "https://legulegu.com/stockdata/shenzhenPE",
            "创业板": "https://legulegu.com/stockdata/cybPE",
        }
        params = {"token": token, "marketId": symbol_map[symbol]}
        r = requests.get(url, params=params, **get_cookie_csrf(url=url_map[symbol]))
        data_r=r.text
        data_json = orjson.loads(data_r)
        temp_df = pd.DataFrame(data_json["data"])
        temp_df["date"] = (
            pd.to_datetime(temp_df["date"], unit="ms", utc=True)
            .dt.tz_convert("Asia/Shanghai")
            .dt.date
        )
        temp_df = temp_df[
            [
                "date",
                "close",
                "pe",
            ]
        ]
        temp_df.columns = [
            "日期",
            "指数",
            "平均市盈率",
        ]
        return temp_df
    else:
        url = "https://legulegu.com/api/stockdata/get-ke-chuang-ban-pe"
        params = {"token": token}
        r = requests.get(url, params=params, **get_cookie_csrf(url="https://legulegu.com/stockdata/ke-chuang-ban-pe"))
        data_r=r.text
        data_json = orjson.loads(data_r)
        temp_df = pd.DataFrame(data_json["data"])
        temp_df["date"] = (
            pd.to_datetime(temp_df["date"], unit="ms", utc=True)
            .dt.tz_convert("Asia/Shanghai")
            .dt.date
        )
        temp_df = temp_df[
            [
                "date",
                "close",
                "pe",
            ]
        ]
        temp_df.columns = [
            "日期",
            "总市值",
            "市盈率",
        ]
        return temp_df
#1.爬取市场主要指数的收盘价数据
index_list=['000300.SH','000905.SH','000906.SH','000852.SH','000016.SH','000688.SH']
name_list=['沪深300','中证500','上证50','中证1000','中证800','科创50']

@st.cache_data(ttl=12000)
def get_data():
  all_df=pd.DataFrame()
  for i,j in zip(index_list,name_list):
      df = pro.index_daily(ts_code=i,start_date='20200101',fields=[
          "trade_date",
          "close"])
      df.loc[:,'trade_date']=pd.to_datetime(df.loc[:,'trade_date'])
      df=df.sort_values(by='trade_date')
      df=df.rename(columns={'close':j,'trade_date':'date'})
      df=df.set_index('date')
      all_df=pd.concat([all_df,df],axis=1)
  return all_df
#所有指数数据
all_data=get_data()

#2.读取barra因子收益率数据
@st.cache_data(ttl=6000)
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
#barra收益率
barra_factor=load_barra_data()
#名称列表
barra_factor_list =['BarraCNE5_Beta_nav_pct', 'BarraCNE5_BooktoPrice_nav_pct', 'BarraCNE5_DebttoAssets_nav_pct',
    'BarraCNE5_EarningsYield_nav_pct', 'BarraCNE5_Growth_nav_pct', 'BarraCNE5_Liquidity_nav_pct',
    'BarraCNE5_Momentum_nav_pct', 'BarraCNE5_NonLinearSize_nav_pct',
    'BarraCNE5_ResidualVolatility_nav_pct', 'BarraCNE5_Size_nav_pct']
#3.读取期货风格因子收益率数据
@st.cache_data
def load_future_data():
    df=pd.read_csv("期货风格因子日收益率",index_col=0)
    df=df.rename(columns={'price_date':'date'})
    df=df.set_index('date')
    return df
future_data=load_future_data()

def cai_future_return(future_data):
    future_data1=future_data.copy()
    dfff=future_data1.iloc[:,:16].iloc[-22:,:]
    for i in dfff.columns:
        dfff[i+'nav']=1
        dfff[i+'nav'].iloc[1:]=(1+dfff[i].iloc[1:]).cumprod()
        dfff[i+'nav']=dfff[i+'nav']-1
    return dfff.iloc[:,16:]
future_data_return=cai_future_return(future_data)

#4.计算barra因子暴露度
def barra_ana(df,code):
    lf=df.copy()
    b1 = np.array(lf[code])  # 因变量
    A1 = np.array(lf[barra_factor_list])
    def minmean(A1,b1):
        num_x = np.shape(A1)[1]
        def my_func(x):
            ls = np.abs((b1-np.dot(A1,x))**2)
            #ld=lam*np.sum([pow(x[n], 2) for n in range(7, 17)])
            result = np.sum(ls)#+ld
            return result
        def g1(x):
            return np.sum(x) #sum of X >= 0
        def g2(x):
            return 1-np.sum(x) #sum of X = 1
        cons = ({'type': 'ineq', 'fun': g1}
                ,{'type': 'eq', 'fun':  g2})
        x0  = np.array([ -100, -100, -100, -
                100, -100, -100, -100, -100, -100, -100])
        bnds = [ (None, None), (None, None), (None, None), (None, None), (None, None),
            (None, None), (None, None), (None, None), (None, None), (None, None)]
        res = minimize(my_func, 
                    bounds = bnds, x0=x0,
                    constraints=cons)
        return res
    res=minmean(A1,b1)
    beta_1 = res.x[0]  
    beta_2 = res.x[1]
    beta_3 = res.x[2]
    beta_4 = res.x[3]
    beta_5 = res.x[4]
    beta_6 = res.x[5]
    beta_7 = res.x[6]
    beta_8 = res.x[7]
    beta_9 = res.x[8]
    beta_10 = res.x[9]
    
    CIS_Barra =dict()
    CIS_Barra['Beta因子暴露'] = beta_1
    CIS_Barra['账面市值比因子暴露'] = beta_2
    CIS_Barra['盈利预期因子暴露'] = beta_3
    CIS_Barra['成长因子暴露'] = beta_4
    CIS_Barra['杠杆因子暴露'] = beta_5
    CIS_Barra['流动性因子暴露'] = beta_6
    CIS_Barra['动量因子暴露'] = beta_7
    CIS_Barra['非线性市值因子暴露'] = beta_8
    CIS_Barra['残差波动率因子暴露'] = beta_9
    CIS_Barra['市值因子暴露'] = beta_10
    CIS_Barra=pd.DataFrame([CIS_Barra])
    return CIS_Barra,res
#5.读取申万风格指数数据
@st.cache_data(ttl=6000)
def load_sw():
    df=pd.read_csv("sw风格指数数据",index_col=0)
    df.index=pd.DatetimeIndex(df.index)
    return df
sw_style=load_sw()
#6.读取申万以及行业数据
@st.cache_data(ttl=6000)
def load_sw_1():
    df=pd.read_csv("申万一级数据",index_col=0)
    list1=list(df['指数名称'].unique())
    df_list=[df.loc[df['指数名称']==i][['日期','收盘价']].rename(columns={'收盘价':i}).set_index('日期') for i in list1]
    df_all=pd.concat(df_list,axis=1)
    df_all.index=pd.DatetimeIndex(df_all.index)
    return df_all
sw_1=load_sw_1()
#7.读取私募策略指数数据
@st.cache_data
def load_simu_index():
    df=pd.read_csv("私募指数数据",index_col=0)
    list1=list(df['火富牛私募策略指数'].unique())
    df_list=[df.loc[df['火富牛私募策略指数']==i][['price_date','nav']].rename(columns={'nav':i}).set_index('price_date') for i in list1]
    df_all=pd.concat(df_list,axis=1)
    df_all=df_all.reset_index().sort_values(by='price_date').set_index('price_date')
    return df_all
simu_index=load_simu_index()
#获取概念板块主力近5日资金流入
@st.cache_data(ttl=600)
def get_tech_data():
    stock_fund_flow_concept_df = ak.stock_fund_flow_concept(symbol="5日排行")
    df1=stock_fund_flow_concept_df[['行业','净额']].sort_values(by='净额',ascending=False)
    df1.columns=['概念名称','主力净流入']
    return df1
tech_data=get_tech_data()

#获取大盘资金流向
@st.cache_data(ttl=6000)
def get_money_flow():
    url='https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get?lmt=0&klt=101&fields1=f1%2Cf2%2Cf3%2Cf7&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61%2Cf62%2Cf63%2Cf64%2Cf65&ut=b2884a393a59ad64002292a3e90d46a5&secid=1.000001&secid2=0.399001&_=1699334564623'
    r=requests.get(url)
    data_text=r.text
    df=orjson.loads(data_text)['data']['klines']
    data_=pd.DataFrame(items.split(',')  for items in df)
    data_=data_.iloc[:,:6]
    data_.columns=['date','主力净流入','小单净流入','中单净流入','大单净流入','超大单净流入']
    for i in data_.columns[1:]:
        data_[i]=data_[i].apply(lambda x: round(float(x)/100000000.0,3))
    return data_
money=get_money_flow()
money['date']=pd.to_datetime(money['date'])
money=money.set_index('date')
#获取板块今日资金净流入情况
@st.cache_data(ttl=600)
def get_industry():
    stock_fund_flow_industry_df = ak.stock_fund_flow_industry(symbol="5日排行")
    df=stock_fund_flow_industry_df[['行业','净额']].sort_values(by='净额',ascending=False)
    df.columns=['行业名称','资金净流入']
    return df
industry_money=get_industry()

#获取股票主力排名
@st.cache_data(ttl=6000)
def best_data():
    url='https://push2.eastmoney.com/api/qt/clist/get?fid=f184&po=1&pz=50&pn=1&np=1&fltt=2&invt=2&fields=f2%2Cf3%2Cf12%2Cf13%2Cf14%2Cf62%2Cf184%2Cf225%2Cf165%2Cf263%2Cf109%2Cf175%2Cf264%2Cf160%2Cf100%2Cf124%2Cf265%2Cf1&ut=b2884a393a59ad64002292a3e90d46a5&fs=m%3A0%2Bt%3A6%2Bf%3A!2%2Cm%3A0%2Bt%3A13%2Bf%3A!2%2Cm%3A0%2Bt%3A80%2Bf%3A!2%2Cm%3A1%2Bt%3A2%2Bf%3A!2%2Cm%3A1%2Bt%3A23%2Bf%3A!2%2Cm%3A0%2Bt%3A7%2Bf%3A!2%2Cm%3A1%2Bt%3A3%2Bf%3A!2'
    r=requests.get(url)
    data_text=r.text
    df=pd.DataFrame(orjson.loads(data_text)['data']['diff']).iloc[:20,:]
    df1=df[['f12','f14','f100','f2','f225','f263','f264']]
    df1.columns=['股票代码','股票名称','行业','最新价','今日排名','5日排名','10日排名']
    df1.loc[:,'最新价']=df1.loc[:,'最新价'].apply(lambda x: round(x,2))
    return df1
best_stock_data=best_data()

#北向资金情况
@st.cache_data(ttl=6000)
def north_money():
    stock_hsgt_north_net_flow_in_em_df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
    stock_hsgt_north_net_flow_in_em_df.loc[:,'value']=stock_hsgt_north_net_flow_in_em_df.loc[:,'value'].apply(lambda x: round(x/10000,2))
    stock_hsgt_north_net_flow_in_em_df.columns=['date','北向资金净流入']
    stock_hsgt_north_net_flow_in_em_df=stock_hsgt_north_net_flow_in_em_df.set_index('date')
    stock_hsgt_north_net_flow_in_em_df.index=pd.DatetimeIndex(stock_hsgt_north_net_flow_in_em_df.index)
    return stock_hsgt_north_net_flow_in_em_df
north_money1=north_money()
#获取上证平均市盈率
@st.cache_data(ttl=6000)
def pe():
    stock_market_pe_lg_df = stock_market_pe_lg(symbol="上证")
    stock_market_pe_lg_df=stock_market_pe_lg_df.set_index('日期')
    stock_market_pe_lg_df.index=pd.DatetimeIndex(stock_market_pe_lg_df.index)
    return stock_market_pe_lg_df
pe_=pe()
#获取股债利差
@st.cache_data(ttl=6000)
def stock_bond():
    stock_ebs_lg_df = ak.stock_ebs_lg()
    stock_ebs_lg_df=stock_ebs_lg_df.set_index('日期')
    stock_ebs_lg_df.index=pd.DatetimeIndex(stock_ebs_lg_df.index)
    return stock_ebs_lg_df
stock_bond1=stock_bond()
#获取大盘拥挤度
@st.cache_data(ttl=6000)
def get_cross():
    stock_a_congestion_lg_df = ak.stock_a_congestion_lg()
    stock_a_congestion_lg_df.columns=['date','上证指数','大盘拥挤度']
    stock_a_congestion_lg_df=stock_a_congestion_lg_df.set_index('date')
    stock_a_congestion_lg_df.index=pd.DatetimeIndex(stock_a_congestion_lg_df.index)
    return stock_a_congestion_lg_df
cross=get_cross()
#CPI数据
@st.cache_data(ttl=6000)
def chinese_cpi_index():
    macro_china_cpi_monthly_df = ak.macro_china_cpi_monthly()
    df=pd.DataFrame(macro_china_cpi_monthly_df)
    df.index.name='date'
    return df
cpi=chinese_cpi_index()
#PPI
@st.cache_data(ttl=6000)
def get_ppi():
    macro_china_ppi_yearly_df = ak.macro_china_ppi_yearly()
    df=pd.DataFrame(macro_china_ppi_yearly_df)
    df.index.name='date'
    return df
ppi=get_ppi()
#PMI
@st.cache_data(ttl=6000)
def get_pmi():
    macro_china_pmi_yearly_df = ak.macro_china_pmi_yearly()
    macro_china_non_man_pmi_df = ak.macro_china_non_man_pmi()
    df=pd.concat([macro_china_pmi_yearly_df,macro_china_non_man_pmi_df],axis=1)
    df.index.name='date'
    df.columns=['制造业','非制造业']
    return df
pmi=get_pmi()
#上海间隔夜拆借利率
@st.cache_data(ttl=6000)
def get_shibor():
    r=requests.get('https://datacenter-web.eastmoney.com/api/data/v1/get?reportName=RPT_IMP_INTRESTRATEN&columns=REPORT_DATE%2CREPORT_PERIOD%2CIR_RATE%2CCHANGE_RATE%2CINDICATOR_ID%2CLATEST_RECORD%2CMARKET%2CMARKET_CODE%2CCURRENCY%2CCURRENCY_CODE&quoteColumns=&filter=(LATEST_RECORD%3D1)(MARKET_CODE%3D%22001%22)&pageNumber=1&pageSize=20&sortTypes=1&sortColumns=INDICATOR_ID&source=WEB&client=WEB&_=1699339690131')
    data_text=r.text
    df=pd.DataFrame(orjson.loads(data_text)['result']['data'])
    df=df[['REPORT_DATE','REPORT_PERIOD','IR_RATE','CHANGE_RATE','MARKET']]
    df.columns=['date','品种','利率(%)','涨跌(BP)','市场']
    df['date']=pd.to_datetime(df['date'])
    df['date']=df['date'].apply(lambda x:x.strftime('%Y-%m-%d'))
    return df
shibor=get_shibor()
#定义一个计算barra暴露度的函数
all_data1=all_data.pct_change().fillna(0)
df_all=pd.merge(barra_factor,all_data1,left_index=True,right_index=True,how='left')
#计算指数在barra上的暴露度
def cal_barra(df_all):
    all1=df_all.copy()
    all1=all1.iloc[-252:,]
    ddd=pd.concat([ barra_ana(all1,i)[0] for i in ['沪深300','中证500','中证1000','中证800','上证50','科创50']])
    ddd.insert(0,'指数名称',['沪深300','中证500','中证1000','中证800','上证50','科创50'])
    for i in ddd.columns[1:]:
      ddd[i]=ddd[i].apply(lambda x: round(x,3))
    D=ddd.T
    return D

exposure=cal_barra(df_all)

def cal_barra_return(barra_factor):
    barra_factor1=barra_factor.copy()
    dfff=barra_factor1.iloc[:,:10].iloc[-22:,:]
    for i in dfff.columns:
        dfff[i+'nav']=1
        dfff[i+'nav'].iloc[1:]=(1+dfff[i].iloc[1:]).cumprod()
        dfff[i+'nav']=dfff[i+'nav']-1
    return dfff.iloc[:,10:]

barra_return=cal_barra_return(barra_factor)

#指数的周度涨跌幅
chinese_index_name_list=list(all_data.columns)
week_return=[round(i*100,2) for i in all_data.iloc[-1,]/all_data.iloc[-5,]-1]
#计算申万风格指数的周度涨跌幅
sw_name=list(sw_style.columns)
sw_return=[round(i*100,2) for i in list(sw_style.iloc[-1,:]/sw_style.iloc[-5,:]-1)]
#计算申万一级行业的月度涨跌幅
sw_1_name=list(sw_1.columns)
sw_1_return=[round(i*100,2) for i in list(sw_1.iloc[-1,:]/sw_1.iloc[-22,:]-1)]
#申万一级行业近一年的
sw_1_return1=[round(i*100,2) for i in list(sw_1.iloc[-1,:]/sw_1.iloc[-252,:]-1)]

def RV(x):
    return np.sqrt(np.sum(x**2))

def cai_std(all_data):
    all_data2=all_data.pct_change().fillna(0)
    index1=all_data2.groupby(pd.Grouper(freq='5d')).apply(RV).iloc[-22:,]
    index2=all_data2.groupby(pd.Grouper(freq='10d')).apply(RV).iloc[-22:,]
    index3=all_data2.groupby(pd.Grouper(freq='20d')).apply(RV).iloc[-22:,]

    return index1,index2,index3

index=cai_std(all_data)

with st.container():
   fig = go.Figure(data = (
        go.Bar(x=chinese_index_name_list,  # x轴数据
               y=week_return,text=week_return,textposition="outside" # y轴数据
              )))
   fig.update_layout(title_text='主要指数的近5日绝对收益率:'+all_data.index[-1].strftime('%Y-%m-%d')) 
   st.plotly_chart(fig)

   fi8 = go.Figure()
   d1=[]
   for i in index[0].columns:
       d1.append(go.Scatter(x=index[0].index.strftime('%Y-%m-%d'),y=index[0][i],mode='lines',name=i))
   fi8 = go.Figure(data=d1)

# 柱状图模式需要设置：4选1
   fi8.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5)) 
   fi8.update_xaxes(rangeslider_visible=True,linecolor='#c5c5c5')
   st.markdown('**各指数5日时序波动率**')
   st.plotly_chart(fi8)

   fi9 = go.Figure()
   d2=[]
   for i in index[1].columns:
       d2.append(go.Scatter(x=index[1].index.strftime('%Y-%m-%d'),y=index[1][i],mode='lines',name=i))
   fi9 = go.Figure(data=d2)

# 柱状图模式需要设置：4选1
   fi9.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5)) 
   fi9.update_xaxes(rangeslider_visible=True,linecolor='#c5c5c5')
   st.markdown('**各指数10日时序波动率**')
   st.plotly_chart(fi9)

   fi10 = go.Figure()
   d3=[]
   for i in index[2].columns:
       d3.append(go.Scatter(x=index[2].index.strftime('%Y-%m-%d'),y=index[2][i],mode='lines',name=i))
   fi10 = go.Figure(data=d3)

# 柱状图模式需要设置：4选1
   fi10.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5)) 
   fi10.update_xaxes(rangeslider_visible=True,linecolor='#c5c5c5')
   st.markdown('**各指数20日时序波动率**')
   st.plotly_chart(fi10)


   year_data=all_data.iloc[-252:,:]
   year_data=year_data/year_data.iloc[0,]
   sw_year_data=sw_style.iloc[-252:,:]
   sw_year_data=sw_year_data/sw_year_data.iloc[0,]
   sw_1_year_data=sw_1.iloc[-252:,:]
   sw_1_year_data=sw_1_year_data/sw_1_year_data.iloc[0,]
   #绘制近一年的收盘价
   lines=[]
   for i in year_data.columns:
       lines.append(go.Scatter(x=year_data.index.strftime('%Y-%m-%d'),y=list(year_data[i]),name=i))
   fig4 = go.Figure(data=lines)
   fig4.update_layout(title_text='指数近一年的收盘价走势(归一化处理):'+year_data.index[-1].strftime('%Y-%m-%d'),legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5))
   fig4.update_xaxes(rangeslider_visible=True,linecolor='#c5c5c5')
   st.plotly_chart(fig4)

   fig1 = go.Figure(data = (
        go.Bar(x=sw_name,  # x轴数据
               y=sw_return,text=sw_return,textposition="outside" # y轴数据
              )))
   fig1.update_layout(title_text='申万风格指数近5日绝对收益率:'+sw_style.index[-1].strftime('%Y-%m-%d')) 
   st.plotly_chart(fig1)
   lines1=[]
   for i in sw_year_data.columns:
       lines1.append(go.Scatter(x=sw_year_data.index.strftime('%Y-%m-%d'),y=list(sw_year_data[i]),name=i))
   fig5 = go.Figure(data=lines1)
   fig5.update_layout(title_text='申万风格指数近一年的收盘价走势(归一化处理):'+sw_year_data.index[-1].strftime('%Y-%m-%d'),legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5))
   fig5.update_xaxes(rangeslider_visible=True,linecolor='#c5c5c5')
   st.plotly_chart(fig5)

   fig2 = go.Figure(data = (
        go.Bar(x=sw_1_name,  # x轴数据
               y=sw_1_return,text=sw_1_return,textposition="outside" # y轴数据
              )))
   fig2.update_layout(title_text='申万一级行业指数的近一月绝对收益率:'+sw_1.index[-1].strftime('%Y-%m-%d')) 
   st.plotly_chart(fig2)

   fig3 = go.Figure(data = (
        go.Bar(x=sw_1_name,  # x轴数据
               y=sw_1_return1,text=sw_1_return1,textposition="outside" # y轴数据
              )))
   fig3.update_layout(title_text='申万一级行业指数的近一年绝对收益率:'+sw_1.index[-1].strftime('%Y-%m-%d')) 
   st.plotly_chart(fig3)

   lines2=[]
   for i in sw_1_year_data.columns:
       lines2.append(go.Scatter(x=sw_1_year_data.index.strftime('%Y-%m-%d'),y=list(sw_1_year_data[i]),name=i))
   fig6 = go.Figure(data=lines2)
   fig6.update_layout(legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5))
   fig6.update_xaxes(rangeslider_visible=True,linecolor='#c5c5c5')
   st.plotly_chart(fig6)


   fi= go.Figure(data = (
        go.Bar(x=list(tech_data['概念名称'].iloc[:10])+list(tech_data['概念名称'].iloc[-10:]),  # x轴数据
                y=list(tech_data['主力净流入'].iloc[:10])+list(tech_data['主力净流入'].iloc[-10:]),
                text=list(tech_data['主力净流入'].iloc[:10])+list(tech_data['主力净流入'].iloc[-10:]),
                textposition="outside" ,marker_color='#af0010'
                )))
   fi.update_layout(title_text='近5日概念主力净流入状况前10+后10,单位(亿)') 
   st.plotly_chart(fi)
   
   fi2= go.Figure(data = (
        go.Bar(x=list(industry_money['行业名称'].iloc[:10])+list(industry_money['行业名称'].iloc[-10:]),  # x轴数据
                y=list(industry_money['资金净流入'].iloc[:10])+list(industry_money['资金净流入'].iloc[-10:]),
                text=list(industry_money['资金净流入'].iloc[:10])+list(industry_money['资金净流入'].iloc[-10:]),
                textposition="outside" ,marker_color='#af0010'
                )))
   fi2.update_layout(title_text='今日日行业主力净流入状况前10+后10,单位(亿)') 
   st.plotly_chart(fi2)

   trace1 = go.Scatter(
   x=cross.index.strftime('%Y-%m-%d'),
   y=list(cross['上证指数']),
   name='上证指数'
    )
   trace2 = go.Scatter(
        x=cross.index.strftime('%Y-%m-%d'),
        y=list(cross['大盘拥挤度']),
        name='大盘拥挤度(右轴)',
        xaxis='x', 
        yaxis='y2'#标明设置一个不同于trace1的一个坐标轴
    )
    
   trace = [trace1, trace2]
   layout = go.Layout(legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5),
        yaxis2=dict(anchor='x', overlaying='y', side='right')#设置坐标轴的格式，一般次坐标轴在右侧
    )
    
   fi17 = go.Figure(data=trace, layout=layout)
   fi17.update_xaxes(rangeslider_visible=True,linecolor='#c5c5c5')
   st.markdown('上证指数及大盘拥挤度')
   st.plotly_chart(fi17)

   trace4 = go.Scatter(
   x=stock_bond1.index.strftime('%Y-%m-%d'),
   y=list(stock_bond1['沪深300指数']),
   name='沪深300指数'
     )
   trace5 = go.Scatter(
    x=stock_bond1.index.strftime('%Y-%m-%d'),
    y=list(stock_bond1['股债利差']),
    name='股债利差(右轴)',
    xaxis='x', 
    yaxis='y2'#标明设置一个不同于trace1的一个坐标轴
     )
   trace6 = go.Scatter(
    x=stock_bond1.index.strftime('%Y-%m-%d'),
    y=list(stock_bond1['股债利差均线']),
    name='股债利差均线(右轴)',
    xaxis='x', 
    yaxis='y2'#标明设置一个不同于trace1的一个坐标轴
      )
   data123 = [trace4, trace5,trace6]
   layout = go.Layout(legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5),
    yaxis2=dict(anchor='x', overlaying='y', side='right')#设置坐标轴的格式，一般次坐标轴在右侧
            )
 
   fi18= go.Figure(data=data123, layout=layout)
   fi18.update_xaxes(rangeslider_visible=True,linecolor='#c5c5c5')
   st.markdown('沪深300及股债利差')
   st.plotly_chart(fi18)

   trace7 = go.Scatter(
   x=pe_.index.strftime('%Y-%m-%d'),
   y=list(pe_['指数']),
   name='上证指数'
    )
   trace8 = go.Scatter(
   x=pe_.index.strftime('%Y-%m-%d'),
   y=list(pe_['平均市盈率']),
    name='平均市盈率(右轴)',
    xaxis='x', 
    yaxis='y2'#标明设置一个不同于trace1的一个坐标轴
    )

   data_1 = [trace7, trace8]
   layout = go.Layout(legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5),
    yaxis2=dict(anchor='x', overlaying='y', side='right')#设置坐标轴的格式，一般次坐标轴在右侧
)
 
   fi20 = go.Figure(data=data_1, layout=layout)
   fi20.update_xaxes(rangeslider_visible=True,linecolor='#c5c5c5')
   st.markdown('上证指数及平均市盈率')
   st.plotly_chart(fi20)

   
   lines3=[]
   for i in money.columns:
            
        lines3.append(go.Scatter(x=money.index.strftime('%Y-%m-%d'),y=list(money[i]),name=i))
   fi1 = go.Figure(data=lines3)
   fi1.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5),title_text='资金流入状况:'+money.index[-1].strftime('%Y-%m-%d'))
   fi1.update_xaxes(rangeslider_visible=True,linecolor='#c5c5c5')
   st.plotly_chart(fi1)
   
   
   fi3=go.Figure(data=[go.Scatter(x=north_money1.index.strftime('%Y-%m-%d'),y=list(north_money1['北向资金净流入']),name='北向资金净流入')])
   fi3.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5),title_text='北向资金流入状况:'+north_money1.index[-1].strftime('%Y-%m-%d')+'单位(亿元)')
   fi3.update_xaxes(rangeslider_visible=True,linecolor='#c5c5c5')
   st.plotly_chart(fi3)
   
   st.markdown('**今日股票排名**:'+north_money1.index[-1].strftime('%Y-%m-%d'))
   table=create_table(best_stock_data)
   st.plotly_chart(table)

   categories = ['Beta因子暴露',
    '账面市值比因子暴露',
    '盈利预期因子暴露',
    '成长因子暴露',
    '杠杆因子暴露',
    '流动性因子暴露',
    '动量因子暴露',
    '非线性市值因子暴露',
    '残差波动率因子暴露',
    '市值因子暴露'] 
   data__=[]
   for i,j in zip(['沪深300','中证500','中证1000','中证800','上证50','科创50'],range(7)):
    data__.append(go.Scatterpolar(name=i,theta=categories,r=list(exposure.iloc[:,j].values[1:]),fill='toself'))
   fi41 = go.Figure(data=data__)
   fi41.update_layout(polar=dict(
                            radialaxis=dict(
                            visible=True
                            )),
                        title_text='指数barra因子暴露度') 
   st.plotly_chart(fi41) 

   all_d=list(all_data.iloc[-1,]/all_data.iloc[-252,]-1)
   all__=list(all_data.columns)

   exposure1=exposure.iloc[1:,]
   exposure1.columns=exposure.iloc[0,:]
   exposure1=exposure.iloc[1:,]
   exposure1.columns=exposure.iloc[0,:]
   exposure1=exposure1.reset_index().rename(columns={'index':'因子暴露度'})
   exposure1=exposure1.set_index('因子暴露度')

   for i,j in zip(all_d,all__):
       exposure1[j]=exposure1[j]*i
   exposure1=exposure1.applymap(lambda x: round(x*100,3))
   dgf=pd.DataFrame()
   dgf['因子暴露度']=[i[:-2] for i in categories]
   for i in all__:
       dgf[i]=list(exposure1[i])
   table1=create_table(dgf)
   st.markdown('近一年各指数在barra因子暴露上的收益率')
   st.plotly_chart(table1)
   
   categorie = ['Beta因子',
    '账面市值比因子',
    '盈利预期因子',
    '成长因子',
    '杠杆因子',
    '流动性因子',
    '动量因子',
    '非线性市值因子',
    '残差波动率因子',
    '市值因子'] 
   fi5 = go.Figure()
   data11=[]
   for i,j in zip(barra_return.columns,categorie):
    data11.append(go.Scatter(x=barra_return.index.strftime('%Y-%m-%d'),y=barra_return[i],mode='lines',name=j))
   fi5 = go.Figure(data=data11)

# 柱状图模式需要设置：4选1
   fi5.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5)) 
   st.markdown('**近一月以来barra因子的累计收益率**')
   st.plotly_chart(fi5)




   future_data_return.index=pd.DatetimeIndex(future_data_return.index)

   fi6 = go.Figure()
   data12=[]
   for i,j in zip(future_data_return.columns,list(future_data.columns)):
    data12.append(go.Scatter(x=future_data_return.index.strftime('%Y-%m-%d'),y=future_data_return[i],mode='lines',name=j))
   fi6 = go.Figure(data=data12)

# 柱状图模式需要设置：4选1
   fi6.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5)) 
   st.markdown('**近一月以来期货风格因子的累计收益率**')
   st.plotly_chart(fi6)

   fi13 = go.Figure()
   d13=[]
   for i in cpi.columns:
    d13.append(go.Scatter(x=cpi.index.strftime('%Y-%m-%d'),y=cpi[i],mode='lines',name=i))
   fi13 = go.Figure(data=d13)

# 柱状图模式需要设置：4选1
   fi13.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5)) 
   fi13.update_xaxes(rangeslider_visible=True,linecolor='#c5c5c5')
   st.markdown('**中国 CPI 月率报告**')
   st.plotly_chart(fi13)

   fi14 = go.Figure()
   d14=[]
   for i in pmi.columns:
    d14.append(go.Scatter(x=pmi.index.strftime('%Y-%m-%d'),y=pmi[i],mode='lines',name=i))
   fi14 = go.Figure(data=d14)

# 柱状图模式需要设置：4选1
   fi14.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5)) 
   fi14.update_xaxes(rangeslider_visible=True,linecolor='#c5c5c5')
   st.markdown('**中国年度PMI数据**')
   st.plotly_chart(fi14)

   fi15 = go.Figure()
   d15=[]
   for i in ppi.columns:
    d15.append(go.Scatter(x=ppi.index.strftime('%Y-%m-%d'),y=ppi[i],mode='lines',name=i))
   fi15 = go.Figure(data=d15)

# 柱状图模式需要设置：4选1
   fi15.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5)) 
   fi15.update_xaxes(rangeslider_visible=True,linecolor='#c5c5c5')
   st.markdown('**中国 PPI 年率报告**')
   st.plotly_chart(fi15)


   table2=create_table(shibor)
   st.markdown('最新shibor')
   st.plotly_chart(table2)

   simu_index1=simu_index.iloc[-252:,]/simu_index.iloc[-252,]
   simu_index1.index=pd.DatetimeIndex(simu_index1.index)
   fi16 = go.Figure()
   d16=[]
   for i in simu_index1.columns:
    d16.append(go.Scatter(x=simu_index1.index.strftime('%Y-%m-%d'),y=simu_index1[i],mode='lines',name=i))
   fi16 = go.Figure(data=d16)

# 柱状图模式需要设置：4选1
   fi16.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5)) 
   st.markdown('**私募策略指数(归一化处理)**')
   st.plotly_chart(fi16)





