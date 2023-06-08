# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
import utils
import eda_app
import seaborn as sns
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

@st.cache_data
def eda_features_date(train, test, transactions, stores, oil, holidays):
    """
    Perform data preprocessing steps for EDA (Exploratory Data Analysis) in the eda_app.

    Args:
        train (pd.DataFrame): Training data.
        test (pd.DataFrame): Test data.
        transactions (pd.DataFrame): Transaction data.
        stores (pd.DataFrame): Store data.
        oil (pd.DataFrame): Oil data.
        holidays (pd.DataFrame): Holiday data.

    Returns:
        tuple: A tuple containing the preprocessed dataframes:
            train (pd.DataFrame): Preprocessed training data.
            test (pd.DataFrame): Preprocessed test data.
            transactions (pd.DataFrame): Preprocessed transaction data.
            stores (pd.DataFrame): Preprocessed store data.
            oil (pd.DataFrame): Preprocessed oil data.
            holidays (pd.DataFrame): Preprocessed holiday data.
            d (pd.DataFrame): Feature-engineered holiday data.

    """
    train["date"] = pd.to_datetime(train.date)
    test["date"] = pd.to_datetime(test.date)
    transactions["date"] = pd.to_datetime(transactions.date)
    oil["date"] = pd.to_datetime(oil.date)
    holidays["date"] = pd.to_datetime(holidays.date)

    train.onpromotion = train.onpromotion.astype("float16")
    train.sales = train.sales.astype("float32")
    stores.cluster = stores.cluster.astype("int8")

    oil = oil.set_index("date").dcoilwtico.resample("D").sum().reset_index()
    oil["dcoilwtico"] = np.where(oil["dcoilwtico"] == 0, np.nan, oil["dcoilwtico"])
    oil["dcoilwtico_interpolated"] = oil.dcoilwtico.interpolate()

    train = train[~((train.store_nbr == 52) & (train.date < "2017-04-20"))]
    train = train[~((train.store_nbr == 22) & (train.date < "2015-10-09"))]
    train = train[~((train.store_nbr == 42) & (train.date < "2015-08-21"))]
    train = train[~((train.store_nbr == 21) & (train.date < "2015-07-24"))]
    train = train[~((train.store_nbr == 29) & (train.date < "2015-03-20"))]
    train = train[~((train.store_nbr == 20) & (train.date < "2015-02-13"))]
    train = train[~((train.store_nbr == 53) & (train.date < "2014-05-29"))]
    train = train[~((train.store_nbr == 36) & (train.date < "2013-05-09"))]

    c = train.groupby(["store_nbr", "family"]).sales.sum().reset_index().sort_values(["family", "store_nbr"])
    c = c[c.sales == 0]

    outer_join = train.merge(c[c.sales == 0].drop("sales", axis=1), how="outer", indicator=True)
    train = outer_join[~(outer_join._merge == "both")].drop("_merge", axis=1)

    d = eda_app.Feature_Engineering_Holidays(holidays, train, test, stores)

    return train, test, transactions, stores, oil, holidays, d

def create_date_features(df):
    """
        Creates additional date-related features from a DataFrame column named "date".

        Args:
            df (pandas.DataFrame): The DataFrame containing the "date" column.

        Returns:
            pandas.DataFrame: The DataFrame with added date-related features.

        Features Added:
            - month: The month of the date (integer).
            - day_of_month: The day of the month (integer).
            - day_of_year: The day of the year (integer).
            - week_of_month: The week of the month (integer).
            - day_of_week: The day of the week (integer, where Monday is 1 and Sunday is 7).
            - year: The year (integer).
            - is_wknd: Whether the date is a weekend (1 for weekend, 0 for weekday).
            - quarter: The quarter of the year (integer).
            - is_month_start: Whether the date is the start of the month (1 for start, 0 otherwise).
            - is_month_end: Whether the date is the end of the month (1 for end, 0 otherwise).
            - is_quarter_start: Whether the date is the start of the quarter (1 for start, 0 otherwise).
            - is_quarter_end: Whether the date is the end of the quarter (1 for end, 0 otherwise).
            - is_year_start: Whether the date is the start of the year (1 for start, 0 otherwise).
            - is_year_end: Whether the date is the end of the year (1 for end, 0 otherwise).
            - season: The season of the year (0 for winter, 1 for spring, 2 for summer, 3 for fall).
    """
    df["month"] = df.date.dt.month.astype("int8")
    df["day_of_month"] = df.date.dt.day.astype("int8")
    df["day_of_year"] = df.date.dt.dayofyear.astype("int16")
    df["week_of_month"] = (df.date.apply(lambda d: (d.day-1)//7 + 1)).astype("int8")
    # df["week_of_year"] = df.date.dt.weekofyear.astype("int8")
    df["day_of_week"] = (df.date.dt.dayofweek + 1).astype("int8")
    df["year"] = df.date.dt.year.astype("int32")
    df["is_wknd"] = (df.date.dt.weekday//4).astype("int8")
    df["quarter"] = df.date.dt.quarter.astype("int8")
    df["is_month_start"] = df.date.dt.is_month_start.astype("int8")
    df["is_month_end"] = df.date.dt.is_month_end.astype("int8")
    df["is_quarter_start"] = df.date.dt.is_quarter_start.astype("int8")
    df["is_quarter_end"] = df.date.dt.is_quarter_end.astype("int8")
    df["is_year_start"] = df.date.dt.is_year_start.astype("int8")
    df["is_year_end"] = df.date.dt.is_year_end.astype("int8")

    # 0 : Winter , 1 : Spring , 2 : Summer , 3 : Fall
    df["season"] = np.where(df.month.isin([12, 1, 2]), 0, 1)
    df["season"] = np.where(df.month.isin([6, 7, 8]), 2, df["season"])
    df["season"] = pd.Series(np.where(df.month.isin([9, 10, 11]), 3, df["season"])).astype("int8")

    return df

def ewm_features(dataframe, alphas, lags):
    """
    Calculates exponential weighted moving averages (EWMA) and returns the modified DataFrame.

    Args:
        dataframe (pandas.DataFrame): The DataFrame containing the data.
        alphas (list): A list of alpha values for EWMA calculation.
        lags (list): A list of lag values for shifting the data.

    Returns:
        pandas.DataFrame: The DataFrame with added EWMA features.

    Features Added:
        - sales_ewm_alpha_{alpha}_lag_{lag}: The exponential weighted moving average of the "sales" column
          for each combination of "store_nbr" and "family" columns, with the specified alpha and lag values.

    """
    dataframe = dataframe.copy()
    for alpha in alphas:
        for lag in lags:
            dataframe["sales_ewm_alpha_" + str(alpha).replace(".","") + "_lag_" + str(lag)] = \
            dataframe.groupby(["store_nbr", "family"])["sales"].\
            transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

def plot_acf_pacf(a,acf_pacf_data):
    """
    Plots ACF (Auto Correlation Function) and PACF (Partial Auto Correlation Function) for a specific family.

    Args:
        a (pandas.DataFrame): The DataFrame containing the data.
        acf_pacf_data (str): The family for which ACF and PACF plots will be generated.

    Returns:
        None

    Plots:
        - ACF plot: Auto Correlation Function plot for the specified family.
        - PACF plot: Partial Auto Correlation Function plot for the specified family.

    """
    try:
        fig, ax = plt.subplots(1, 2, figsize=(15, 4))
        temp = a[(a.family == acf_pacf_data)]
        sm.graphics.tsa.plot_acf(temp.sales, lags=363, ax=ax[0], title="Auto Correlation\n" + acf_pacf_data)
        sm.graphics.tsa.plot_pacf(temp.sales, lags=363, ax=ax[1], title="Partial Auto Correlation\n" + acf_pacf_data)
        st.pyplot(fig)
    except:
        pass


def fig_average_sales(a):
    """
    Plots a graph comparing the average sales across years.

    Args:
        a (pandas.DataFrame): The DataFrame containing the data.

    Returns:
        None

    Plots:
        - Line graph: Average sales comparison across years, with the x-axis representing the day of the year,
          the y-axis representing the sales, and different colors representing different years.

    """
    fig = px.line(a, x="day_of_year", y="sales", color="year")
    fig.update_layout(
        width=1180,
        height=500
    )

    st.plotly_chart(fig)

def fig_SMA_graph(a, store_num ,family_name):
    """
    Plots a graph showing the Simple Moving Average (SMA) for a specific family.

    Args:
        a (pandas.DataFrame): The DataFrame containing the data.
        store_num (str): The store number.
        family_name (str): The family name for which SMA graph will be generated.

    Returns:
        None

    Plots:
        - Line graphs: Simple Moving Average (SMA) comparison for the specified family,
          with different SMAs plotted against the actual sales data.

    """
    # for i in a.family.unique():
    fig, ax = plt.subplots(2, 4, figsize=(20, 10))

    a[a.family == family_name][["sales", "SMA20_sales_lag16"]].plot(legend=True, ax=ax[0, 0], linewidth=4)
    a[a.family == family_name][["sales", "SMA30_sales_lag16"]].plot(legend=True, ax=ax[0, 1], linewidth=4)
    a[a.family == family_name][["sales", "SMA45_sales_lag16"]].plot(legend=True, ax=ax[0, 2], linewidth=4)
    a[a.family == family_name][["sales", "SMA60_sales_lag16"]].plot(legend=True, ax=ax[0, 3], linewidth=4)
    a[a.family == family_name][["sales", "SMA90_sales_lag16"]].plot(legend=True, ax=ax[1, 0], linewidth=4)
    a[a.family == family_name][["sales", "SMA120_sales_lag16"]].plot(legend=True, ax=ax[1, 1], linewidth=4)
    a[a.family == family_name][["sales", "SMA365_sales_lag16"]].plot(legend=True, ax=ax[1, 2], linewidth=4)

    plt.suptitle(f"{store_num} - " + family_name, fontsize=15)
    plt.tight_layout(pad=1.5)
    for j in range(0, 4):
        ax[0, j].legend(fontsize="x-large")
        ax[1, j].legend(fontsize="x-large")

    st.pyplot(fig)

def fig_EMA_graph(a, store_num, family_name):
    """
    Plots a graph showing the Exponential Moving Average (EMA) for a specific family.

    Args:
        a (pandas.DataFrame): The DataFrame containing the data.
        store_num (str): The store number.
        family_name (str): The family name for which EMA graph will be generated.

    Returns:
        None

    Plots:
        - Line graph: Exponential Moving Average (EMA) comparison for the specified family,
          with EMA plotted against the actual sales data.

    """
    fig, ax = plt.subplots(1,1, figsize=(20,10))
    a[(a.store_nbr == store_num) & (a.family == family_name)].set_index("date")[["sales", "sales_ewm_alpha_095_lag_16"]].plot(title=f"{store_num} - {family_name}", ax=ax)
    st.pyplot(fig)

def grouped(df, key, freq, col):
    """
    Groups data based on a certain frequency.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        key (str or list): The column(s) to use as the grouping key(s).
        freq (str or pandas offset alias): The frequency at which to group the data.
        col (str): The column for which to calculate the mean.

    Returns:
        pandas.DataFrame: The grouped DataFrame with the mean value of the specified column.

    """
    df_grouped = df.groupby([pd.Grouper(key=key, freq=freq)]).agg(mean = (col, 'mean'))
    df_grouped = df_grouped.reset_index()
    return df_grouped

def predict_seasonality(df, key, freq, col, ax1, title1):
    """
    Predicts seasonality in data using a linear regression model with Fourier terms.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        key (str or list): The column(s) to use as the grouping key(s) for grouping the data.
        freq (str or pandas offset alias): The frequency at which to group the data.
        col (str): The column containing the target variable for which seasonality will be predicted.
        ax1 (matplotlib.axes.Axes): The matplotlib axes to use for plotting.
        title1 (str): The title for the plot.

    Returns:
        None

    Plots:
        - Line plot: Original mean values, predicted seasonal values, and seasonal forecast.

    """
    fourier = CalendarFourier(freq="A", order=10)  # 10 sin/cos pairs for "A"nnual seasonality
    df_grouped = grouped(df, key, freq, col)
    df_grouped['date'] = pd.to_datetime(df_grouped['date'], format="%Y-%m-%d")
    df_grouped['date'].freq = freq  # manually set the frequency of the index
    dp = DeterministicProcess(index=df_grouped['date'],
                              constant=True,
                              order=1,
                              period=None,
                              seasonal=True,
                              additional_terms=[fourier],
                              drop=True)
    dp.index.freq = freq  # manually set the frequency of the index

    # 'in_sample' creates features for the dates given in the `index` argument
    X1 = dp.in_sample()
    y1 = df_grouped["mean"]  # the target
    y1.index = X1.index

    # The intercept is the same as the `const` feature from
    # DeterministicProcess. LinearRegression behaves badly with duplicated
    # features, so we need to be sure to exclude it here.
    model = LinearRegression(fit_intercept=False)
    model.fit(X1, y1)
    y1_pred = pd.Series(model.predict(X1), index=X1.index)
    X1_fore = dp.out_of_sample(steps=90)
    y1_fore = pd.Series(model.predict(X1_fore), index=X1_fore.index)

    ax1 = y1.plot(linestyle='dashed', style='.', label="init mean values", color="0.4", ax=ax1, use_index=True)
    ax1 = y1_pred.plot(linewidth=3, label="Seasonal", color='b', ax=ax1, use_index=True)
    ax1 = y1_fore.plot(linewidth=3, label="Seasonal Forecast", color='r', ax=ax1, use_index=True)
    ax1.set_title(title1, fontsize=18)
    _ = ax1.legend()

def Seasonal_Forecast(train, trans):
    """
    Plots seasonal forecast graphs for transaction volume and sales.

    Args:
        train (pandas.DataFrame): The DataFrame containing the sales data.
        trans (pandas.DataFrame): The DataFrame containing the transaction data.

    Returns:
        None

    Plots:
        - Line plots: Seasonal forecast for transaction volume and sales.

    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30,8))
    predict_seasonality(trans, 'date', 'W', 'transactions', axes[0], "Transactions Seasonal Forecast")
    predict_seasonality(train, 'date', 'W', 'sales', axes[1], "Sales Seasonal Forecast")

    st.pyplot(fig)


def plot_deterministic_process(df, key, freq, col, ax1, title1, ax2, title2):
    """
    Plots trend forecast graphs.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        key (str): The column name used for grouping.
        freq (str): The frequency of the data (e.g., 'D' for daily, 'M' for monthly).
        col (str): The column for which trend will be predicted.
        ax1 (matplotlib.axes.Axes): The first subplot axes.
        title1 (str): The title for the first subplot.
        ax2 (matplotlib.axes.Axes): The second subplot axes.
        title2 (str): The title for the second subplot.

    Returns:
        None

    Plots:
        - Line plots: Trend forecast and predicted trend.

    """
    df_grouped = grouped(df, key, freq, col)
    df_grouped['date'] = pd.to_datetime(df_grouped['date'], format="%Y-%m-%d")
    dp = DeterministicProcess(index=df_grouped['date'], constant=True, order=1, drop=True)
    dp.index.freq = freq  # manually set the frequency of the index
    # 'in_sample' creates features for the dates given in the `index` argument
    X1 = dp.in_sample()
    y1 = df_grouped["mean"]  # the target
    y1.index = X1.index
    # The intercept is the same as the `const` feature from
    # DeterministicProcess. LinearRegression behaves badly with duplicated
    # features, so we need to be sure to exclude it here.
    model = LinearRegression(fit_intercept=False)
    model.fit(X1, y1)
    y1_pred = pd.Series(model.predict(X1), index=X1.index)
    ax1 = y1.plot(linestyle='dashed', label="mean", color="0.75", ax=ax1, use_index=True)
    ax1 = y1_pred.plot(linewidth=3, label="Trend", color='b', ax=ax1, use_index=True)
    ax1.set_title(title1, fontsize=18)
    _ = ax1.legend()

    # forecast Trend for future 30 steps
    steps = 30
    X2 = dp.out_of_sample(steps=steps)
    y2_fore = pd.Series(model.predict(X2), index=X2.index)
    y2_fore.head()
    ax2 = y1.plot(linestyle='dashed', label="mean", color="0.75", ax=ax2, use_index=True)
    ax2 = y1_pred.plot(linewidth=3, label="Trend", color='b', ax=ax2, use_index=True)
    ax2 = y2_fore.plot(linewidth=3, label="Predicted Trend", color='r', ax=ax2, use_index=True)
    ax2.set_title(title2, fontsize=18)
    _ = ax2.legend()


def Trend_Forecasting(train, trans):
    """
    Plots trend forecasting graphs for transactions and sales.

    Args:
        train (pandas.DataFrame): The DataFrame containing sales data.
        trans (pandas.DataFrame): The DataFrame containing transactions data.

    Returns:
        None

    Plots:
        - Line plots: Linear trend forecast and predicted trend for transactions and sales.

    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(30,12))
    plot_deterministic_process(trans, 'date', 'W', 'transactions',
                               axes[0,0], "Transactions Linear Trend",
                               axes[0,1], "Transactions Linear Trend Forecast")
    plot_deterministic_process(train, 'date', 'W', 'sales',
                               axes[1,0], "Sales Linear Trend",
                               axes[1,1], "Sales Linear Trend Forecast")
    st.pyplot(fig)


def make_lags(ts, lags):
    """
       Generate lagged values for prediction models.

       Parameters:
           ts (pandas.Series): Time series data.
           lags (int): Number of lagged values to generate.

       Returns:
           pandas.DataFrame: DataFrame with lagged values.

       Example:
           >>> ts = pd.Series([1, 2, 3, 4, 5])
           >>> make_lags(ts, 2)
              y_lag_1  y_lag_2
           0      NaN      NaN
           1      1.0      NaN
           2      2.0      1.0
           3      3.0      2.0
           4      4.0      3.0
    """
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(1, lags + 1)
        },
        axis=1)

def lags_forcasting(train, family):
    """
    Applies regression model for forecasting by splitting the train data into validation data.

    Args:
        train (pandas.DataFrame): The DataFrame containing the train data.
        family (str): The family name for which the forecasting is performed.

    Returns:
        None

    Modifies:
        - Updates the 'sales_deseasoned' column in the 'train' DataFrame.

    """
    store_sales = train.copy()
    store_sales['date'] = store_sales.date.dt.to_period('D')
    store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()

    family_sales = (
        store_sales
        .groupby(['family', 'date'])
        .mean()
        .unstack('family')
        .loc['2016', ['sales', 'onpromotion']]
    )

    mag_sales = family_sales.loc(axis=1)[:, family]
    y = mag_sales.loc[:, 'sales'].squeeze()

    fourier = CalendarFourier(freq='M', order=4)
    dp = DeterministicProcess(
        constant=True,
        index=y.index,
        order=1,
        seasonal=True,
        drop=True,
        additional_terms=[fourier],
    )
    X_time = dp.in_sample()
    X_time['NewYearsDay'] = (X_time.index.dayofyear == 1)

    model = LinearRegression(fit_intercept=False)
    model.fit(X_time, y)
    y_deseason = y - model.predict(X_time)
    y_deseason.name = 'sales_deseasoned'

    X = make_lags(y_deseason, lags=4)
    X = X.fillna(0.0)
    y = y_deseason.copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=60, shuffle=False)

    # Fit and predict
    model = LinearRegression()  # `fit_intercept=True` since we didn't use DeterministicProcess
    model.fit(X_train, y_train)
    y_pred = pd.Series(model.predict(X_train), index=y_train.index)
    y_fore = pd.Series(model.predict(X_test), index=y_test.index)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 8))
    ax = y_train.plot(color="0.75", style=".-", markeredgecolor="0.25", markerfacecolor="0.25", ax=ax)
    ax = y_test.plot(color="0.75", style=".-", markeredgecolor="0.25", markerfacecolor="0.25", ax=ax)
    ax = y_pred.plot(ax=ax)
    _ = y_fore.plot(ax=ax, color='C3')
    ax.set_ylabel('Sales', fontsize = 10)
    ax.set_title(f'{family} Forecasting', fontsize = 15)
    st.pyplot(fig)


def stat_app():
    """
        Statistical Analysis App

        This function displays various statistical analyses and visualizations based on the loaded data.
        It includes correlation analysis, ACF/PACF plots, forecasting techniques, and moving average calculations.
    """
    st.write('---')
    # load_data & features_data
    train, test, transactions, stores, oil, holidays = utils.load_data()
    train, test, transactions, stores, oil, holidays, d = eda_features_date(train, test, transactions, stores, oil, holidays)

    selected_data = st.radio('SELECT DATA', ["Correlation", "ACF / PACF", "Forecasting", "Moving Average"])

    # tab1 = 개념설명 / tab2 = 데이터 시각화
    tab1,tab2 = st.tabs(["**Concept**", "**Data**"])

    if selected_data == 'Correlation':
        with tab1:
            st.subheader('What is correlation?')
            st.markdown("Correlation refers to the statistical relationship between two entities. "
                        "In other words, it's how two variables move in relation to one another. Correlation can be used for various data sets, as well. In some cases, you might have predicted how things will correlate, while in others, the relationship will be a surprise to you. It's important to understand that correlation does not mean the relationship is causal.")
            st.markdown("To understand how correlation works, it's important to understand the following terms:")
            st.markdown('- **Positive correlation**: A positive correlation would be 1. This means the two variables moved either up or down in the same direction together.')
            st.markdown('- **Negative correlation**: A negative correlation is -1. This means the two variables moved in opposite directions.')
            st.markdown('- **Zero or no correlation**: A correlation of zero means there is no relationship between the two variables. In other words, as one variable moves one way, the other moved in another unrelated direction.')
            st.image(utils.e_img4,width=1000)
            st.markdown('<br>', unsafe_allow_html=True)
            st.subheader('Types of correlation coefficients')
            st.markdown('While correlation studies how two entities relate to one another, a correlation coefficient measures the strength of the relationship between the two variables. In statistics, there are three types of correlation coefficients. They are as follows:')
            st.markdown("- **Pearson correlation**: The Pearson correlation is the most commonly used measurement for a linear relationship between two variables. The stronger the correlation between these two datasets, the closer it'll be to +1 or -1.")
            st.markdown("- **Spearman correlation**: This type of correlation is used to determine the monotonic relationship or association between two datasets. Unlike the Pearson correlation coefficient, it's based on the ranked values for each dataset and uses skewed or ordinal variables rather than normally distributed ones.")
            st.markdown("- **Kendall correlation**: This type of correlation measures the strength of dependence between two datasets.")
            st.markdown("Knowing your variables is helpful in determining which correlation coefficient type you will use. Using the right correlation equation will help you to better understand the relationship between the datasets you're analyzing.")
            st.markdown("References : https://www.indeed.com/career-advice/career-development/correlation-definition-and-examples")
            
        with tab2:
            # Simple Moving Average (단순 이동 평균)
            a = train.sort_values(["store_nbr", "family", "date"])
            for i in [20, 30, 45, 60, 90, 120, 365]:
                a["SMA" + str(i) + "_sales_lag16"] = a.groupby(["store_nbr", "family"]).rolling(i).sales.mean().shift(
                    16).values
                a["SMA" + str(i) + "_sales_lag30"] = a.groupby(["store_nbr", "family"]).rolling(i).sales.mean().shift(
                    30).values
                a["SMA" + str(i) + "_sales_lag60"] = a.groupby(["store_nbr", "family"]).rolling(i).sales.mean().shift(
                    60).values
            # 상관관계 계산
            corr_matrix = a[["sales"] + a.columns[a.columns.str.startswith("SMA")].tolist()].corr()

            # Heatmap 그리기
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, annot_kws={"fontsize": 7})

            # 부제
            st.subheader('SMA Lags Correlation')

            # Streamlit에 표시
            st.pyplot(fig)

    if selected_data == "ACF / PACF":
        with tab1:
            st.subheader('What is Autocorrelation Function (ACF)?')
            st.markdown("The autocorrelation function (ACF) is a statistical technique that we can use to identify how correlated the values in a time series are with each other. The ACF plots the correlation coefficient against the lag, which is measured in terms of a number of periods or units. A lag corresponds to a certain point in time after which we observe the first value in the time series."
                        "<br>The correlation coefficient can range from -1 (a perfect negative relationship) to +1 (a perfect positive relationship). A coefficient of 0 means that there is no relationship between the variables. Also, most often, it is measured either by Pearson’s correlation coefficient or by Spearman’s rank correlation coefficient. "
                        "<br>It’s most often used to analyze sequences of numbers from random processes, such as economic or scientific measurements. It can also be used to detect systematic patterns in correlated data sets such as securities prices or climate measurements."
                        "<br>Below, we can see an example of the ACF plot:", unsafe_allow_html=True)
            st.image(utils.e_img5, width=1000)
            st.markdown("Blue bars on an ACF plot above are the error bands, and anything within these bars is not statistically significant. It means that correlation values outside of this area are very likely a correlation and not a statistical fluke. The confidence interval is set to 95% by default. "
                        "<br>Notice that for a lag zero, ACF is always equal to one, which makes sense because the signal is always perfectly correlated with itself."
                        "<br>**To summarize, autocorrelation is the correlation between a time series (signal) and a delayed version of itself, while the ACF plots the correlation coefficient against the lag, and it’s a visual representation of autocorrelation.**", unsafe_allow_html=True)
            st.markdown('<br>', unsafe_allow_html=True)
            st.subheader('What is Partial Autocorrelation Function (PACF)?')
            st.markdown("PACF is a partial auto-correlation function. Basically instead of finding correlations of present with lags like ACF, it finds correlation of the residuals (which remains after removing the effects which are already explained by the earlier lag(s)) with the next lag value hence ‘partial’ and not ‘complete’ as we remove already found variations before we find the next correlation. So if there is any hidden information in the residual which can be modeled by the next lag, we might get a good correlation and we will keep that next lag as a feature while modeling. Remember while modeling we don’t want to keep too many features which are correlated as that can create multicollinearity issues. Hence we need to retain only the relevant features."
                        "<br>The figure below presents the PACF plot:", unsafe_allow_html=True)
            st.image(utils.e_img6, width=1000)
            st.markdown("**To summarize, a partial autocorrelation function captures a “direct” correlation between time series and a lagged version of itself.**")
            st.markdown("References : https://www.baeldung.com/cs/acf-pacf-plots-arma-modeling")

        with tab2:
            # Time Related Features
            d = create_date_features(d)

            # Workday column
            d["workday"] = np.where(
                (d.holiday_national_binary == 1) | (d.holiday_local_binary == 1) | (d.holiday_regional_binary == 1) | (
                    d['day_of_week'].isin([6, 7])), 0, 1)
            d["workday"] = pd.Series(np.where(d.IsWorkDay.notnull(), 1, d["workday"])).astype("int8")
            d.drop("IsWorkDay", axis=1, inplace=True)

            # Wages in the public sector are paid every two weeks on the 15th and on the last day of the month.
            # Supermarket sales could be affected by this.
            d["wageday"] = pd.Series(np.where((d['is_month_end'] == 1) | (d["day_of_month"] == 15), 1, 0)).astype(
                "int8")

            # ACF & PACF
            a = d[(d.sales.notnull())].groupby(["date", "family"]).sales.mean().reset_index().set_index("date")

            ## family 에 대한 sales  정보를 ACF 와 PACF 상관관계
            st.subheader('ACF / PACF Plot')
            family_name = st.selectbox(' ', a['family'].unique(), format_func=lambda x: f"Family: {x}")
            plot_acf_pacf(a, family_name)
            st.markdown('<br>', unsafe_allow_html=True)
            st.subheader('Average Sales for 2015 and 2016')
            a = d[d.year.isin([2015, 2016])].groupby(["year", "day_of_year"]).sales.mean().reset_index()

            ## 연도에 따라 판매 평균을 비교
            fig_average_sales(a)
            st.markdown('**ACF / PACF 그래프를 확인해 본 후 PACF에서 16, 20, 30, 45, 365, 730 Lag(지연)을 선택하기로 결정했습니다. '
                        '특히 모델을 개선하는 데에는 365일과 730일 Lag(지연)이 도움이 될 수 있을 것으로 생각됩니다. <br>'
                        '이에 따라 Lag-365를 적용. 2015년과 2016년의 매출을 비교해 본 결과 상관관계가 높다는 것을 확인할 수 있고, Lag-730인 2017년 또한 비슷할 것으로 추측됩니다.**', unsafe_allow_html=True)

    if selected_data == "Forecasting":
        with tab1:
            st.subheader('What Is Forecasting?')
            st.markdown('Forecasting is a technique that uses historical data as inputs to make informed estimates that are predictive in determining the direction of future trends.')
            st.markdown('**<span style="font-size:25px;">Key Takeaways</span>**', unsafe_allow_html=True)
            st.markdown('- Forecasting involves making predictions about the future.')
            st.markdown('- In finance, forecasting is used by companies to estimate earnings or other data for subsequent periods.')
            st.markdown('- Traders and analysts use forecasts in valuation models, to time trades, and to identify trends.')
            st.markdown('Forecasts are often predicated on historical data. Because the future is uncertain, forecasts must often be revised, and actual results can vary greatly.')
            st.markdown('<br>', unsafe_allow_html=True)
            st.subheader('Forecasting Techniques')
            st.markdown('In general, forecasting can be approached using qualitative techniques or quantitative ones. '
                        'Quantitative methods of forecasting exclude expert opinions and utilize statistical data based on quantitative information. '
                        'Quantitative forecasting models include time series methods, discounting, analysis of leading or lagging indicators, and econometric modeling that may try to ascertain causal links.')
            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown('**<span style="font-size:25px;"> Time Series Forecasting </span>**', unsafe_allow_html=True)
            st.markdown('Time series forecasting is the process of analyzing time series data using statistics and modeling to make predictions and inform strategic decision-making. '
                        'It’s not always an exact prediction, and likelihood of forecasts can vary wildly—especially when dealing with the commonly fluctuating variables in time series data as well as factors outside our control. '
                        '<br>However, forecasting insight about which outcomes are more likely—or less likely—to occur than other potential outcomes. Often, the more comprehensive the data we have, the more accurate the forecasts can be. '
                        'While forecasting and “prediction” generally mean the same thing, there is a notable distinction. In some industries, forecasting might refer to data at a specific future point in time, while prediction refers to future data in general. '
                        'Series forecasting is often used in conjunction with time series analysis. Time series analysis involves developing models to gain an understanding of the data to understand the underlying causes. Analysis can provide the “why” behind the outcomes you are seeing. '
                        'Forecasting then takes the next step of what to do with that knowledge and the predictable extrapolations of what might happen in the future. The Box-Jenkins Model is a technique designed to forecast data ranges based on inputs from a specified time series. '
                        '<br>It forecasts data using three principles: autoregression, differencing, and moving averages. Another method, known as rescaled range analysis, can be used to detect and evaluate the amount of persistence, randomness, or mean reversion in time series data. '
                        'The rescaled range can be used to extrapolate a future value or average for the data to see if a trend is stable or likely to reverse.', unsafe_allow_html=True)
            st.markdown('Most often, time series forecasts involve trend analysis, cyclical fluctuation analysis, and issues of seasonality.')
            st.markdown('Reference : https://www.investopedia.com/terms/m/movingaverage.asp')

        with tab2:
            st.subheader('Seasonal Forecasting')
            Seasonal_Forecast(train, transactions)
            st.markdown('<br>', unsafe_allow_html=True)
            st.subheader('Trend Forecasting')
            Trend_Forecasting(train, transactions)
            st.markdown('<br>', unsafe_allow_html=True)
            st.subheader('Lags Forecasting')

            # 지연 예측 그래프를 family별로 선택해서 볼 수 있도록 selectbox 생성
            family_name = st.selectbox(' ', train['family'].unique(), format_func=lambda x: f"Family: {x}")
            lags_forcasting(train, family_name)


    if selected_data == 'Moving Average':
        with tab1:
            st.subheader('What Is a Moving Average (MA)?')
            st.markdown('In finance, a moving average (MA) is a stock indicator commonly used in technical analysis. The reason for calculating the moving average of a stock is to help smooth out the price data by creating a constantly updated average price.')
            st.markdown('By calculating the moving average, the impacts of random, short-term fluctuations on the price of a stock over a specified time frame are mitigated. Simple moving averages (SMAs) use a simple arithmetic average of prices over some timespan, while exponential moving averages (EMAs) place greater weight on more recent prices than older ones over the time period.')
            st.markdown('**<span style="font-size:25px;">Key Takeaways</span>**', unsafe_allow_html=True)
            st.markdown('- A moving average (MA) is a stock indicator commonly used in technical analysis.')
            st.markdown('- The moving average helps to level the price data over a specified period by creating a constantly updated average price.')
            st.markdown('- A simple moving average (SMA) is a calculation that takes the arithmetic mean of a given set of prices over a specific number of days in the past.')
            st.markdown('- An exponential moving average (EMA) is a weighted average that gives greater importance to the price of a stock in more recent days, making it an indicator that is more responsive to new information.')
            st.markdown('<br>', unsafe_allow_html=True)
            st.subheader('Simple Moving Average (SMA)?')
            st.markdown('A simple moving average (SMA), is calculated by taking the arithmetic mean of a given set of values over a specified period. A set of numbers, or prices of stocks, are added together and then divided by the number of prices in the set.'
                        '<br>>The formula for calculating the simple moving average of a security is as follows:')
            st.markdown('(A1 + A2 + A3 + A4…An) / n = SMA Where n is the number of time periods and A is the average within a given time period.')
            st.markdown('<br>', unsafe_allow_html=True)
            st.subheader('Exponential Moving Average (EMA)?')
            st.markdown('The exponential moving average gives more weight to recent prices in an attempt to make them more responsive to new information. To calculate an EMA, the simple moving average (SMA) over a particular period is calculated first.')
            st.markdown('Then calculate the multiplier for weighting the EMA, known as the "smoothing factor," which typically follows the formula: [2/(selected time period + 1)]. ')
            st.markdown('For a 20-day moving average, the multiplier would be [2/(20+1)]= 0.0952. The smoothing factor is combined with the previous EMA to arrive at the current value. The EMA thus gives a higher weighting to recent prices, while the SMA assigns an equal weighting to all values.')
            st.markdown('<br>', unsafe_allow_html=True)
            st.subheader('Simple Moving Average (SMA) vs. Exponential Moving Average (EMA)')
            st.markdown('The calculation for EMA puts more emphasis on the recent data points. Because of this, EMA is considered a weighted average calculation.')
            st.markdown('In the figure below, the number of periods used in each average is 15, but the EMA responds more quickly to the changing prices than the SMA. The EMA has a higher value when the price is rising than the SMA and it falls faster than the SMA when the price is declining. This responsiveness to price changes is the main reason why some traders prefer to use the EMA over the SMA.')
            st.markdown('<br>', unsafe_allow_html=True)
            st.image(utils.e_img7, width=600)
            st.markdown('Reference : https://www.investopedia.com/terms/m/movingaverage.asp')

        with tab2:
            # Simple Moving Average (단순 이동 평균)
            a = train.sort_values(["store_nbr", "family", "date"])
            for i in [20, 30, 45, 60, 90, 120, 365]:
                a["SMA" + str(i) + "_sales_lag16"] = a.groupby(["store_nbr", "family"]).rolling(i).sales.mean().shift(
                    16).values
                a["SMA" + str(i) + "_sales_lag30"] = a.groupby(["store_nbr", "family"]).rolling(i).sales.mean().shift(
                    30).values
                a["SMA" + str(i) + "_sales_lag60"] = a.groupby(["store_nbr", "family"]).rolling(i).sales.mean().shift(
                    60).values

            col1, col2 = st.columns([5, 5])
            # 그래프를 store별 family별로 선택해서 확인할 수 있도록 selectbox 생성
            with col1:
                store_num = st.selectbox(' ', a['store_nbr'].unique(), format_func=lambda x: f"store_nbr: {x}")
            with col2:
                family_name = st.selectbox(' ', a['family'].unique(), format_func=lambda x: f"Family: {x}")
            b = a[(a.store_nbr == store_num)].set_index("date")

            ## 단순이동평균을 보여주는 그래프
            st.subheader('Simple Moving Average(SMA)')
            fig_SMA_graph(b, store_num, family_name)

            # 가중치
            alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
            # 지연
            lags = [16, 30, 60, 90]

            a = ewm_features(a, alphas, lags)

            ## 지수평균이동을 보여주는 그래프
            st.subheader('Exponential Moving Average(EMA)')
            fig_EMA_graph(a, store_num, family_name)