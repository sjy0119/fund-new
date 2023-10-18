import tushare as ts
pro = ts.pro_api('8e812052c92d7a829f0e3b0197d248e48bb2ba3efbbaa60f505e6852')
import streamlit as st
import akshare as ak
from datetime import date
import plotly.express as px
st.set_page_config(page_icon="🌼",)
st.markdown("# 基金主体信息数据查询")
st.sidebar.header("基金主体信息数据查询")
st.write(
    """在该模块之中，大家可以选择输入基金代码便可以获得该基金的主体信息和行业配置信息，仅显示占净值比例大于1%的行业"""
)
code=st.text_input('请输入基金代码例如000001.OF')
if code:
    df = pro.fund_basic(**{
    "ts_code":code,
    "market": "",
    "update_flag": "",
    "offset": "",
    "limit": "",
    "status": "",
    "name": ""
}, fields=[
    "ts_code",
    "name",
    "management",
    "custodian",
    "fund_type",
    "found_date",
    "due_date",
    "list_date",
    "issue_date",
    "delist_date",
    "issue_amount",
    "m_fee",
    "c_fee",
    "duration_year",
    "p_value",
    "min_amount",
    "exp_return",
    "benchmark",
    "status",
    "invest_type",
    "type",
    "trustee",
    "purc_startdate",
    "redm_startdate",
    "market"
])
    df1=df.T
    df1.columns=['基本信息']
    ts_code=code[:6]
    dt=int(str(date.today())[:4])

    if code:
        try:
            fund_portfolio_industry_allocation_em_df = ak.fund_portfolio_industry_allocation_em(symbol=ts_code, date=str(dt))
        except:
            fund_portfolio_industry_allocation_em_df = ak.fund_portfolio_industry_allocation_em(symbol=ts_code, date=str(dt-1))
        re_fund=fund_portfolio_industry_allocation_em_df.loc[fund_portfolio_industry_allocation_em_df['截止时间']==fund_portfolio_industry_allocation_em_df['截止时间'][0]]
        new_fund=re_fund.loc[re_fund['占净值比例']>=1]
        fig= px.bar(new_fund,x='行业类别',y='占净值比例')
    if st.checkbox('显示信息'):
        st.dataframe(df1,width=500)
    if st.checkbox('展示基金行业配置信息'):
        st.plotly_chart(fig)

