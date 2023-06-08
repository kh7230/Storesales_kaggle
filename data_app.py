# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import utils


def summary(dataframe):
    """
       Display the dataframe, data types, and describe statistics in a Streamlit-style format.

       :param dataframe: The input dataframe.
       :return: None
    """
    # st.set_page_config(page_title="Store Sales", page_icon=":💰:",
    #                         layout = "wide", initial_sidebar_state="expanded")
    # 화면 분할을 위한 컬럼 설정 2:1 비율
    col1, col2 = st.columns([4, 4])
    with col1:
        st.title("📣 Data")
        st.dataframe(dataframe, height=810, width=1200)

    with col2:
        st.title("📣 Data Type")
        st.dataframe(dataframe.dtypes, height=350, width=650)

        st.title("📣 Describe")
        st.dataframe(dataframe.describe(), height=350, width=650)

def data_app():
    """
        Display data descriptions and summaries based on the selected data type.

        :return: None
    """
    train, test, transactions, stores, oil, holidays = utils.load_data()

    # 데이터 딕셔너리 생성
    datalist_dict = {
        "✓ Train": train,
        "✓ Test": test,
        "✓ Transactions": transactions,
        "✓ Stores": stores,
        "✓ Oil": oil,
        "✓ Holidays_Events": holidays
    }

    # selectbox 생성
    datalist = st.selectbox("✅ 데이터 종류", list(datalist_dict.keys()), index=0)
    st.markdown("---")
    st.subheader(f"📝{datalist} Data Description")

    # 조건문으로 요약 정보 넣기
    if datalist == "✓ Train":
        st.markdown("""- **:red[Train Data]** 는 **:green[상점 번호, 제품군, 프로모션 및 목표 매출]** 로 구성된 시계열 데이터입니다.""")
        st.markdown("- **:red[store_nbr]** 은 제품이 판매되는 **:green[상점 번호]** 를 나타냅니다.")
        st.markdown("- **:red[family]** 는 판매되는 **:green[제품 유형]** 을 나타냅니다.")
        st.markdown("- **:red[sales]** 는 특정 날짜에 특정 가게에서 **:green[판매되는 제품군의 총 매출]** 을 나타냅니다. (일부 제품은 소수점 단위로 판매될 수 있으므로 분수 값이 가능합니다.)")
        st.markdown("- **:red[onpromotion]** 은 특정 날짜에 상점에서 **:green[프로모션 중인 제품군의 항목 수]** 를 나타냅니다.")
        summary(train)

    elif datalist == "✓ Test":
        st.markdown("- 학습 데이터와 동일한 기능을 가지는 테스트 데이터입니다. 이 파일의 날짜에 대한 **:red[목표 매출을 예측]** 할 것입니다.")
        st.markdown("- 테스트 데이터의 날짜는 학습 데이터의 마지막 날짜 이후 **:red[15일]** 동안입니다.")
        summary(test)

    elif datalist == "✓ Transactions":
        st.markdown("- 트랜잭션 데이터는 **:red[train 데이터의 sale 열과 높은 상관 관계]** 가 있습니다. 해당 데이터를 통해 매장의 **:green[판매 패턴을 파악]** 할 수 있습니다.")
        summary(transactions)

    elif datalist == "✓ Stores":
        st.markdown("- **:red[도시, 주, 유형, 클러스터 등]** 스토어에 대한 데이터입니다.")
        summary(stores)

    elif datalist == "✓ Oil":
        st.markdown("- 일일 유가, 학습 및 테스트 데이터 기간 등 **:red[전체값을 포함]** 합니다.")
        st.markdown("- 에콰도르는 석유 의존국이며, **:red[석유 가격 변동에 매우 민감]** 합니다.")
        summary(oil)

    elif datalist == "✓ Holidays_Events":
        st.markdown("- 메타데이터와 함께 **:red[휴일 및 이벤트 정보가 포함]** 된 파일입니다.")
        st.markdown("""- **참고** : transferred 열에 주목해야 합니다. 공식적으로 해당 날짜가 공휴일이지만 정부에 의해서 휴일이 옮겨지면 이전된 날짜는
                        휴일보다는 평일에 가깝습니다. 그렇기 때문에 **:red[실제 공휴일을 찾으려면 유형]** 이 **:green[transfer]** 인 행 을 찾으면됩니다. 예를들어
                        Independencia de Guayaquil의 휴일은 2012-10-09에서 2012-10-12로 이전된 사례가 있었는데, 이는 2012-10-12가 휴일로 바뀌었음을 의미합니다.
                        행 유형이 **:green[bridge]** 는 **:red[휴일에 추가로 부여되는 휴일]** 입니다. (예 : 금토일) 한국을 기준으로 말하자면
                        대체 공휴일과 같은 개념입니다.
                    """)
        st.markdown("- 추가적인 휴일 지정으로는 크리스마스 이브 공휴일 지정 같은 것 등이 있습니다.")
        summary(holidays)
