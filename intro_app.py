# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import utils

def intro_app():
    """
        Renders the introduction section of the app, including tabs for overview, objectives, and analysis phases.
    """
    tab1, tab2, tab3 = st.tabs(["**소개**", "**목표**", "**분석 단계**"])
    with tab1:
        st.subheader(":white_check_mark: 대회 개요")
        st.write("")
        st.markdown("대회에서 제공된 데이터는 ***:blue[Corporación Favorita]*** 라는 에콰도르의 식료품 소매 업체의 데이터 입니다.")
        st.write("")
        st.write("")

        col1, col2, col3 = st.columns([3, 5, 2])
        with col1:
            st.write("")
        with col2:
            st.image(utils.e_img1, width=280)
            st.image(utils.e_img2, width=280)
        with col3:
            st.write("")

        st.write("")
        st.write("")
        st.write(
            "**Supermaxi** 대형 유통 체인을 운영하고 있는 기업으로 잘 알려진 **:violet[Corporación Favorita]** 는 에콰도르에서 활동하고 있는 기업들 중 **:red[매출액 1위]** 를 유지하고 있는 유망한 기업입니다.")
        st.write(
            "**Corporación Favorita** 은 남미의 다른 국가에서도 사업을 운영하고 있으며 총 **:green[54개의 Corporación Favorita 의 지점과 33개의 제품에 관한 데이터를 통해 앞으로의 매출액을 예측할 예정]** 입니다.")
        st.write("그리고 데이터 분석을 위해 제공된 **Corporación Favorita** 의 데이터는 **2015-01-01 ~ 2016-12-31** 까지의 데이터입니다.")


    with tab2:
        st.markdown('### :white_check_mark: 대회 목표 \n'
                    '- 이번 대회의 목표는 **:red[시계열 예측]** 을 사용하여 에콰도르에 본사를 두고 있는 대형 식료품 소매업체인 \"**Corporación Favorita**\"의 데이터를 분석하고 매장의 앞으로의 매출을 예측하는 것입니다.\n\n'
                    '- 구체적으로는 여러 Favorita 매장에서 판매되는 수많은 품목의 판매 단가를 보다 **:red[정확하게 예측하는 모델을 구축하는 것이 최종 목표]** 입니다.\n\n'
                    '- 날짜, 매장 및 품목 정보, 프로모션, 판매 단가로 구성된 비교적 접근성이 좋은 학습 데이터 셋을 통해 머신러닝 모델들을 연습할 수도 있습니다.\n\n')
        st.markdown("---")
        st.markdown("### :white_check_mark: 평가 \n"
                    "- 이 대회의 평가 지표는 ***:violet[Root Mean Squared Logarithmic Error]*** (평균 제곱근 오차)입니다. \n")
        st.latex(r'''
            {RMSLE} = \sqrt{ \frac{1}{n} \sum_{i=1}^n \left(\log (1 + \hat{y}_i) - \log (1 + y_i)\right)^2}
            ''')
        st.markdown(
                    "- $n$ n은 **:green[총 인스턴스의 수]** 입니다. \n"
                    "- $\hat{y}_i$ i는 인스턴스 i에 대한 **:green[예측된 타겟 값]** 입니다. \n"
                    "- $y_i$ 는 인스턴스 i에 대한 **:green[실제 타겟]** 입니다. \n"
                    "- $\log$ 는 **:green[자연 로그]** 입니다. \n"
                    )
        st.markdown("---")
        st.markdown("### :bookmark_tabs: 대회 정보 \n"
                    "**More Detailed : [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)**")

    with tab3:
        st.markdown("<h2><center> &#x1F4DD; 머신러닝 4단계 </center></h2>", unsafe_allow_html=True)
        st.write("")
        st.write("")

        col1,col2,col3 = st.columns([1.7,6.3,2])
        with col1:
            st.write(' ')
        with col2:
            st.image(utils.e_img3, width=1000)
        with col3:
            st.write(' ')
        # st.image(utils.e_img3, use_column_width=True)

        st.write("")
        st.write("")
        st.write("")
        st.markdown("---")

        selected_item = st.selectbox(":white_check_mark: 머신러닝 4단계", ['✔ 문제 이해', '✔ 탐색적 데이터 분석', '✔ 베이스 라인 모델', '✔ 성능 개선 및 검증'])

        if selected_item == '✔ 문제 이해':
            st.markdown("- 어떤 문제든 주어진 **:red[문제를 이해]** 하면서 시작해야 합니다. 문제를 정확하게 이해해야 원하는 목표지점을 정확하게 설정할 수 있습니다.")
            st.markdown("- 평가 지표에 대한 이해가 부족하다면 같은 모델을 사용해도 낮은 평가를 받을 수 있습니다.")

        elif selected_item == '✔ 탐색적 데이터 분석':
            st.markdown("- 머신러닝은 데이터를 다루는 기술이기 때문에 데이터를 잘 알아야 가장 효과"
                        "적인 모델을 찾고 최적화할 수 있으므로 **:red[최우선적으로 주어진 데이터들을 면밀하게 분석]** 합니다.")
            st.markdown("- 라이브러리를 사용해서 다양한 그래프를 그려보고 어떤 피처가 중요한지, 어떻게 피처들을 조합해서 새로운 피처를 만들지 등을 통해 **:red[인사이트를 도출합니다 .]**")

        elif selected_item == '✔ 베이스 라인 모델':
            st.markdown("- 간단한 모델을 생성 및 훈련 시켜 **:red[성능을 확인하는 작업]** 을 진행합니다.(기본적인 머신러닝 파이프라인 - 필요에 따라 피처 엔지니어링 진행) ")
            st.markdown("- 베이스 라인 모델 단계에서는 성능은 신경 쓰지 않고 **:red[Train 데이터로 훈련]** 을 시킨다.")
            st.markdown("- 훈련된 모델을 **:red[Test 데이터]** 를 활용해서 결과를 예측해 제출합니다.")
            st.markdown("- 베이스라인 모델의 예측 결과도 같이 제출하는데 그 이유는 **:red[성능이 어느정도 개선되었는지 확인해보기 위함]** 입니다.")

        elif selected_item == '✔ 성능 개선 및 검증':
            st.markdown("- 해당 단계는 성능 개선 및 검증 단계로 **:red[이상치 제거, 결측값 처리, 데이터 인코딩, 피처 스케일링 등을 수행]** 하고 모델 최적의 하이퍼 파라미터를 찾는 작업을 진행합니다.")
            st.markdown("- 성능 검증을 진행 할 때는 훈련화 모델의 일반화 성능을 먼저 평가하고 **:red[성능에 문제가 있다면 문제 이해나 탐색적 데이터 분석 또는 피처 엔지니어링부터 다시 수행]** 합니다.")
            st.markdown("- 성능 평가에 훈련 데이터를 여러 그룹으로 나누어 일부는 훈련 시 사용하고, 일부는 검증 시 사용해서 모델 성능을 측정하는 기법인 **:red[교차 검증]** 을 사용하기도 합니다.")
            st.markdown("- 위의 모든 과정을 마친 뒤 최종적으로 **:red[개선된 모델을 활용해 결과를 예측하고 제출]** 합니다.")
