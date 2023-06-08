# Kaggle Store Sales - 시계열 예측 (~ 2023.05.17)
<br/>

## [1. Streamlit 웹 서비스](https://kh7230-storesales-kaggle-main-app-rr55kv.streamlit.app// "Streamlit Link")<br/>

## [2. 개인 ipynb 파일](https://github.com/kh7230/Storesales_kaggle/blob/main/pdf/storesales.ipynb/ ".ipynb Link")<br/>

## [3. PDF 파일](https://github.com/kh7230/Storesales_kaggle/blob/main/pdf/StoreSales_Attention.pdf/ "PDF Link")<br/>

---
![image](https://github.com/kh7230/Storesales_kaggle/blob/main/img/main-store.png)

## ✔ 목적
Kaggle Store Sales 대회에서 시계열 데이터를 사용하여 Corporación Favorita라는 대규모 에콰도르 **식료품 소매업체 매출 예측** 및 Streamlit 활용 **웹 서비스 구현**<br/>

### [Kaggle Store Sales 대회](https://www.kaggle.com/competitions/store-sales-time-series-forecasting, "Kaggle Link") <br/>

## ✔ 데이터
Store Sales 대회에서 제공하는 데이터를 사용하였습니다.<br/>

### [Store Sales Data](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data, "Data Link") <br/>

## ✔ ERD
![image](https://github.com/kh7230/Storesales_kaggle/blob/main/img/STORESALES_EDR.png)
<br/>

## ✔ 팀 구성
- 사용언어 : Python 3.10.10
- 작업툴 : VS Code
- 인원 : 4명
- 주요 업무 : Streamlit 라이브러리 활용 웹 서비스 구현 및 머신러닝을 활용한 매장 매출 예측
- 기간 : 2023.05.01 ~ 2023.05.17

## ✔ 주요 기능
- **INTRO 페이지**
  - Store Sales 대회에 대한 소개, 목표, 분석 단계에 대한 설명
- **DATA 페이지**
  - Store Sales 대회에서 제공하는 데이터에 대한 정보와 기술통계량 제공

![image1](https://github.com/kh7230/Storesales_kaggle/blob/main/img/github-data.png)
- **Exploratory Data Analysis**
  - 시간 지남에 따른 상점별 매출 시각화
  - 시간 지남에 따른 제품군별 매출 시각화
  - 주/월별 지연값에 대한 매출 시각화
  - 주/월별 매출에 대한 시각화
  - 시간 지남에 따른 상점별 거래량 시각화
  - 월별 거래량 시각화
  - 매출과 거래량 간 상관분석 시각화
  - 요일별 거래량 시각화
  - 시간 지남에 따른 유가 시각화
  - 유가와 매출 및 거래량 간 상관분석 및 시각화
  - 유가와 제품군별 매출 간 상관분석 및 시각화

![image2](https://github.com/kh7230/Storesales_kaggle/blob/main/img/github-eda.png)
- **STAT**
  - 상관분석에 대한 설명 및 지연값들 간 상관분석 시각화
  - ACF / PACF에 대한 설명 및 제품군별 ACF / PACF 시각화
  - Features(추세, 계절성, 지연값)에 대한 설명 및 시각화
  - 주/월 단순이동평균선 및 지수이동평균선 시각화

![image3](https://github.com/kh7230/Storesales_kaggle/blob/main/img/stat.png)
- **ML**
  - RandomForest 모델을 사용한 상점별 매출 및 거래량 예측

![image4](https://github.com/kh7230/Storesales_kaggle/blob/main/img/ML.png)
## ✔ 설치 방법

### Windows
- 버전 확인
  - VS Code : 
  - Python : 3.10.10
  - 라이브러리 : Pandas(1.5.3), NumPy(1.24.3), Math, Seaborn(0.11.2), Streamlit(1.22.0), Matplotlib(3.7.1), Plotly(5.14.1), Scikit-learn(1.2.2), Scipy(1.10.1)

- 프로젝트 파일을 다운로드 받습니다.
```
git clone https://github.com/kh7230/Storesales_kaggle.git
```
- 프로젝트 경로에서 가상환경 설치 후 접속합니다. (Windows 10 기준)
```
virtualenv venv
source venv/Scripts/activate
```
- 라이브러리를 설치합니다.
```
pip install -r requirements.txt
```
- streamlit 명령어를 실행합니다.
```
streamlit run app.py
```
