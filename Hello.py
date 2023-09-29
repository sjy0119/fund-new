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
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
    page_title="基金数据分析",
    page_icon="👋",
)

    
    st.write("# 欢迎来到基金数据分析! 👋")
    st.sidebar.success("选择您要查询的内容")
    
    st.markdown(
            """
           在这个网页之中，您可以查到
    
           1.基金的主体信息
    
           2.基金的净值数据
    
           3.基金的业绩表现
    
           4.最大回撤分析
    
           5.相关性分析
    
           6.基金择股择时能力分析
    
           7.中信风格指数归因
    
           8.中信-BARRA业绩归因
    
           后续将会添加其他内容，敬请期待。。。。。😀
        """
        )
if __name__ == "__main__":
    run()
