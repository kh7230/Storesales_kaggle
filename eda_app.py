# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import shapiro
import scipy.stats as stats
import utils
import plotly.graph_objects as go

def one_hot_encoder(df, nan_as_category=True):
    """
    Perform one-hot encoding on categorical columns in a pandas DataFrame.

    Args:
        df (pandas DataFrame): The input DataFrame.
        nan_as_category (bool, optional): Whether to treat NaN values as a separate category. Defaults to True.

    Returns:
        encoded_df (pandas DataFrame): The DataFrame with one-hot encoded categorical columns.
        encoded_columns (list): The list of column names in the encoded DataFrame.

    Example:
        import pandas as pd

        data = {'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue'],
                'Size': ['S', 'M', 'L', 'M', 'S'],
                'Price': [10, 15, 20, 12, 18]}
        df = pd.DataFrame(data)

        encoded_df, encoded_columns = one_hot_encoder(df)
        print(encoded_df)
        # Output:
        #    Price  Color_Blue  Color_Green  Color_Red  Size_L  Size_M  Size_S
        # 0     10           0            0          1       0       0       1
        # 1     15           1            0          0       0       1       0
        # 2     20           0            1          0       1       0       0
        # 3     12           0            0          1       0       1       0
        # 4     18           1            0          0       0       0       1
    """

    original_columns = list(df.columns)
    categorical_columns = df.select_dtypes(["category", "object"]).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    df.columns = df.columns.str.replace(" ", "_")
    return df, df.columns.tolist()

def AB_Test(dataframe, group, target):
    """
        Perform an A/B test to compare two groups in a dataframe.

        Args:
            dataframe (pandas DataFrame): The input dataframe containing the data.
            group (str): The column name representing the group membership (A/B).
            target (str): The column name representing the target variable to compare between the groups.

        Returns:
            result (pandas DataFrame): A summary of the A/B test results, including hypothesis testing, p-values,
                                       group statistics, and additional information.

        Example:
            import pandas as pd

            data = {'Group': ['A', 'A', 'B', 'B', 'A', 'B'],
                    'Value': [10, 12, 8, 9, 11, 14]}
            df = pd.DataFrame(data)

            result = AB_Test(df, 'Group', 'Value')
            print(result)
            # Output:
            #   Feature     Test Type AB Hypothesis   p-value                  Comment  GroupA_mean  GroupB_mean  GroupA_median  GroupB_median
            # 0   Group  Non-Parametric   Reject H0  0.323810  A/B groups are similar    11.000000     10.333333           11.0            9.0
    """
    # Split A/B
    groupA = dataframe[dataframe[group]==1][target]
    groupB = dataframe[dataframe[group]==0][target]

    if len(groupA) < 3:
        print("Not enough data.")
        return None

    # Assumption: Normality
    ntA = shapiro(groupA)[1] < 0.05
    ntB = shapiro(groupB)[1] < 0.05
    # H0 : Distribution is Normal - False
    # H1 : Distaribution is not Normal - True

    if (ntA == False) & (ntB == False): # H0 : Normal Distribution
        # Parametic Test
        # Assumption: Homogeneity of variances
        leveneTest = stats.levene(groupA, groupB)[1] < 0.05
        # H0 : Homogeneity - False
        # H1 : Heterogeneous: True

        if leveneTest == False:
            # Homogeneity
            ttest = stats.ttest_ind(groupA, groupB, equal_var=True)[1]
            # H0 : M1 == M2 - False
            # H1 : M1 != M2 - True
        else:
            # Heterogeneous
            ttest = stats.ttest_ind(groupA, groupB, equal_var=False)[1]
            # H0 : M1 == M2 - False
            # H1 : M1 != M2 - True
    else:
        # Non-Parametric Test
        ttest = stats.mannwhitneyu(groupA, groupB)[1]
        # H0 : M1 == M2 - False
        # H1 : M1 != M2 - True

    #Result
    temp = pd.DataFrame({
        "AB Hypothesis" : [ttest < 0.05],
        "p-value" : [ttest]
    })
    temp["Test Type"] = np.where((ntA == False) & (ntB == False), "Parametric", "Non-Parametric")
    temp["AB Hypothesis"] = np.where(temp["AB Hypothesis"] == False, "Fail to Reject H0", "Reject H0")
    temp["Comment"] = np.where(temp["AB Hypothesis"] == "Fail to Reject H0", "A/B groups are similar", "A/B groups are not similar")
    temp["Feature"] = group
    temp["GroupA_mean"] = groupA.mean()
    temp["GroupB_mean"] = groupB.mean()
    temp["GroupA_median"] = groupA.median()
    temp["GroupB_median"] = groupB.median()

    # Columns
    if (ntA == False) & (ntB == False):
        temp["Homogeneity"] = np.where(leveneTest == False, "Yes", "No")
        temp = temp[["Feature", "Test Type", "Homogeneity", "AB Hypothesis", "p-value", "Comment", "GroupA_mean", "GroupB_mean", "GroupA_median", "GroupB_median"]]
    else:
        temp = temp[["Feature", "Test Type", "AB Hypothesis", "p-value", "Comment", "GroupA_mean", "GroupB_mean", "GroupA_median", "GroupB_median"]]

    return temp

def fig_Transactions_TotalSales_Correlation(temp, transactions):
    """
    Generate a graph to visualize the correlation pattern between Transactions data and Total Sales.

    Args:
        temp (pandas DataFrame): The DataFrame containing the combined data of Total Sales and Transactions.
        transactions (pandas DataFrame): The DataFrame containing the Transactions data.

    Returns:
        None

    Example:
        import pandas as pd
        import plotly.express as px
        import matplotlib.pyplot as plt
        import streamlit as st

        # Assume 'temp' and 'transactions' DataFrames are defined

        fig_Transactions_TotalSales_Correlation(temp, transactions)
    """
    # temp = pd.merge(train.groupby(["date", "store_nbr"]).sales.sum().reset_index(), transactions, how="left")
    st.write("Spearman Correlation between Total Sales and Transactions: {:,.4f}".format(temp.corr("spearman").sales.loc["transactions"]))

    fig, ax = plt.subplots()
    fig = px.line(transactions.sort_values(["store_nbr", "date"]), x="date", y="transactions", color="store_nbr", title="Transactions")
    fig.update_layout(
        width = 1400,
        height = 600
    )
    st.plotly_chart(fig)

def fig_Transactions_ym_patten1(transactions):
    """
    Generate a graph to visualize the pattern of Transactions data by year and month.

    Args:
        transactions (pandas DataFrame): The DataFrame containing the Transactions data.

    Returns:
        None

    Example:
        import pandas as pd
        import plotly.express as px
        import matplotlib.pyplot as plt
        import streamlit as st

        # Assume 'transactions' DataFrame is defined

        fig_Transactions_ym_patten1(transactions)
    """
    a = transactions.copy()
    a["year"] = a.date.dt.year
    a["month"] = a.date.dt.month

    fig, ax = plt.subplots()
    fig = px.box(a, x="year", y="transactions", color="month", title="Monthly Total Transactions")

    st.plotly_chart(fig)

def fig_Transactions_ym_patten2(transactions):
    """
    Generate a graph to visualize the pattern of average sales by year and month in the Transactions data.

    Args:
        transactions (pandas DataFrame): The DataFrame containing the Transactions data.

    Returns:
        None

    Example:
        import pandas as pd
        import plotly.express as px
        import matplotlib.pyplot as plt
        import streamlit as st

        # Assume 'transactions' DataFrame is defined

        fig_Transactions_ym_patten2(transactions)
    """
    a = transactions.set_index("date").resample("M").transactions.mean().reset_index()
    a["year"] = a.date.dt.year

    fig, ax = plt.subplots()
    fig = px.line(a, x="date", y="transactions", color="year", title="Monthly Average Transactions")

    st.plotly_chart(fig)

def fig_Transactions_Sales_Correlation(temp):
    """
    Generate a graph to visualize the correlation pattern between Transactions data and Sales.

    Args:
        temp (pandas DataFrame): The DataFrame containing the combined data of Sales and Transactions.

    Returns:
        None

    Example:
        import pandas as pd
        import plotly.express as px
        import matplotlib.pyplot as plt
        import streamlit as st

        # Assume 'temp' DataFrame is defined

        fig_Transactions_Sales_Correlation(temp)
    """
    # temp = pd.merge(train.groupby(["date", "store_nbr"]).sales.sum().reset_index(), transactions, how="left")

    fig, ax = plt.subplots()
    fig = px.scatter(temp, x="transactions", y="sales", trendline="ols", trendline_color_override="red")
    fig.update_layout(
        width=1400,
        height=600
    )
    st.plotly_chart(fig)

def fig_Transactions_ydw_patten(transactions):
    """
    Generate a graph to visualize the pattern of Transactions data by year and day of the week.

    Args:
        transactions (pandas DataFrame): The DataFrame containing the Transactions data.

    Returns:
        None

    Example:
        import pandas as pd
        import plotly.express as px
        import matplotlib.pyplot as plt
        import streamlit as st

        # Assume 'transactions' DataFrame is defined

        fig_Transactions_ydw_patten(transactions)
    """
    a = transactions.copy()
    a["year"] = a.date.dt.year
    a["dayofweek"] = a.date.dt.dayofweek + 1
    a = a.groupby(["year", "dayofweek"]).transactions.mean().reset_index()

    fig, ax = plt.subplots()
    fig = px.line(a, x="dayofweek", y="transactions", color="year", title="Transactions Dayofweek")
    fig.update_layout(
        width=1400,
        height=600
    )
    st.plotly_chart(fig)

def fig_OilPrice(oil):
    """
    Generate a graph to visualize the Oil Price data with added missing values.

    Args:
        oil (pandas DataFrame): The DataFrame containing the Oil Price data.

    Returns:
        None

    Example:
        import pandas as pd
        import plotly.express as px
        import matplotlib.pyplot as plt
        import streamlit as st

        # Assume 'oil' DataFrame is defined

        fig_OilPrice(oil)
    """
    # oil = oil.set_index("date").dcoilwtico.resample("D").sum().reset_index()
    # oil["dcoilwtico"] = np.where(oil["dcoilwtico"] == 0, np.nan, oil["dcoilwtico"])
    # oil["dcoilwtico_interpolated"] = oil.dcoilwtico.interpolate()
    p = oil.melt(id_vars=["date"] + list(oil.keys()[5:]), var_name="Legend")

    fig, ax = plt.subplots()
    fig = px.line(p.sort_values(["Legend", "date"], ascending=[False, True]), x="date", y="value", color="Legend", title="Daily Oil Price")
    fig.update_layout(
        width=1400,
        height=600
    )
    st.plotly_chart(fig)


def fig_OilPrice_Sales_Transactions_patten(temp, oil):
    """
    Generate graphs to visualize the patterns between Oil Price and Sales, and Oil Price and Transactions.

    Args:
        temp (pandas DataFrame): The DataFrame containing the combined data of Sales and Transactions.
        oil (pandas DataFrame): The DataFrame containing the Oil Price data.

    Returns:
        None

    Example:
        import pandas as pd
        import matplotlib.pyplot as plt
        import streamlit as st

        # Assume 'temp' and 'oil' DataFrames are defined

        fig_OilPrice_Sales_Transactions_patten(temp, oil)
    """
    # temp = pd.merge(train.groupby(["date", "store_nbr"]).sales.sum().reset_index(), transactions, how="left")
    temp = pd.merge(temp, oil, how="left")
    st.write("Correnlation with Daily Oil Prices")
    st.write(temp.drop(["store_nbr", "dcoilwtico"], axis=1).corr("spearman").dcoilwtico_interpolated.loc[["sales", "transactions"]], "\n")

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    temp.plot.scatter(x="dcoilwtico_interpolated", y="transactions", ax=ax[0])
    temp.plot.scatter(x="dcoilwtico_interpolated", y="sales", ax=ax[1], color="r")
    ax[0].set_title("Daily Oil Price & Transactions", fontsize=15)
    ax[1].set_title("Daily Oil Price & Sales", fontsize=15)
    st.pyplot(fig)

def fig_OilPrice_family_patten(train, oil):
    """
    Generate graphs to visualize the patterns between Oil Price and Sales for each product family.

    Args:
        train (pandas DataFrame): The DataFrame containing the training data with the "date" and "family" columns.
        oil (pandas DataFrame): The DataFrame containing the Oil Price data.

    Returns:
        None

    Example:
        import pandas as pd
        import matplotlib.pyplot as plt
        import streamlit as st

        # Assume 'train' and 'oil' DataFrames are defined

        fig_OilPrice_family_patten(train, oil)
    """
    a = pd.merge(train.groupby(["date", "family"]).sales.sum().reset_index(), oil.drop("dcoilwtico", axis=1), how="left")
    c = a.groupby("family").corr("spearman").reset_index()
    c = c[c.level_1 == "dcoilwtico_interpolated"][["family", "sales"]].sort_values("sales")

    fig, ax = plt.subplots(7, 5, figsize=(20, 20))
    for i, fam in enumerate(c.family):
        if i < 6:
            a[a.family == fam].plot.scatter(x="dcoilwtico_interpolated", y="sales", ax=ax[0, i - 1])
            ax[0, i - 1].set_title(fam + "\n Correlation:" + str(c[c.family == fam].sales.iloc[0])[:6], fontsize=12)
            ax[0, i - 1].axvline(x=70, color="r", linestyle="--")
        if i >= 6 and i < 11:
            a[a.family == fam].plot.scatter(x="dcoilwtico_interpolated", y="sales", ax=ax[1, i - 6])
            ax[1, i - 6].set_title(fam + "\n Correlation:" + str(c[c.family == fam].sales.iloc[0])[:6], fontsize=12)
            ax[1, i - 6].axvline(x=70, color='r', linestyle='--')
        if i >= 11 and i < 16:
            a[a.family == fam].plot.scatter(x="dcoilwtico_interpolated", y="sales", ax=ax[2, i - 11])
            ax[2, i - 11].set_title(fam + "\n Correlation:" + str(c[c.family == fam].sales.iloc[0])[:6], fontsize=12)
            ax[2, i - 11].axvline(x=70, color='r', linestyle='--')
        if i >= 16 and i < 21:
            a[a.family == fam].plot.scatter(x="dcoilwtico_interpolated", y="sales", ax=ax[3, i - 16])
            ax[3, i - 16].set_title(fam + "\n Correlation:" + str(c[c.family == fam].sales.iloc[0])[:6], fontsize=12)
            ax[3, i - 16].axvline(x=70, color='r', linestyle='--')
        if i >= 21 and i < 26:
            a[a.family == fam].plot.scatter(x="dcoilwtico_interpolated", y="sales", ax=ax[4, i - 21])
            ax[4, i - 21].set_title(fam + "\n Correlation:" + str(c[c.family == fam].sales.iloc[0])[:6], fontsize=12)
            ax[4, i - 21].axvline(x=70, color='r', linestyle='--')
        if i >= 26 and i < 31:
            a[a.family == fam].plot.scatter(x="dcoilwtico_interpolated", y="sales", ax=ax[5, i - 26])
            ax[5, i - 26].set_title(fam + "\n Correlation:" + str(c[c.family == fam].sales.iloc[0])[:6], fontsize=12)
            ax[5, i - 26].axvline(x=70, color='r', linestyle='--')
        if i >= 31:
            a[a.family == fam].plot.scatter(x="dcoilwtico_interpolated", y="sales", ax=ax[6, i - 31])
            ax[6, i - 31].set_title(fam + "\n Correlation:" + str(c[c.family == fam].sales.iloc[0])[:6], fontsize=12)
            ax[6, i - 31].axvline(x=70, color='r', linestyle='--')

    plt.tight_layout()
    # plt.suptitle("Daily Oil Product & Total Family Sales \n", fontsize=20)
    st.pyplot(fig)

def fig_Train_sales_Correlation(train):
    """
    Generate a correlation graph for Sales among different stores.

    Args:
        train (pandas DataFrame): The DataFrame containing the training data with the "store_nbr" and "sales" columns.

    Returns:
        None

    Example:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import streamlit as st

        # Assume 'train' DataFrame is defined

        fig_Train_sales_Correlation(train)
    """
    a = train[["store_nbr", "sales"]]
    a["ind"] = 1
    a["ind"] = a.groupby("store_nbr").ind.cumsum().values
    a = pd.pivot(a, index="ind", columns="store_nbr", values="sales").corr()

    mask = np.triu(a.corr())
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    sns.heatmap(a, annot=True, fmt=".1f", cmap="coolwarm", square=True, mask=mask, linewidths=1, cbar=False)
    # plt.title("Correlation among stores", fontsize=20)
    st.pyplot(fig)

def fig_Train_store_TotalSales_patten(train):
    """
    Generate a graph to observe the Total Sales pattern for each store.

    Args:
        train (pandas DataFrame): The DataFrame containing the training data with the "date", "store_nbr", and "sales" columns.

    Returns:
        None

    Example:
        import pandas as pd
        import plotly.express as px
        import matplotlib.pyplot as plt
        import streamlit as st

        # Assume 'train' DataFrame is defined

        fig_Train_store_TotalSales_patten(train)
    """
    a = train.set_index("date").groupby("store_nbr").resample("D").sales.sum().reset_index()

    fig, ax = plt.subplots()
    fig = px.line(a, x="date", y="sales", color="store_nbr", title="Daily Total Sales of The Stores")
    fig.update_layout(
        width=1400,
        height=700
    )
    st.plotly_chart(fig)


def fig_unsold_family(train):
    """
    Generate a graph to identify the unsold product families.

    Args:
        train (pandas DataFrame): The DataFrame containing the training data with the "date", "store_nbr", "family", and "sales" columns.

    Returns:
        None

    Example:
        import pandas as pd
        import matplotlib.pyplot as plt
        import streamlit as st

        # Assume 'train' DataFrame is defined

        fig_unsold_family(train)
    """
    c = train.groupby(["family", "store_nbr"]).tail(60).groupby(["family", "store_nbr"]).sales.sum().reset_index()

    fig, ax = plt.subplots(1, 5, figsize=(20, 4))
    train[(train.store_nbr == 10) & (train.family == "LAWN AND GARDEN")].set_index("date").sales.plot(ax=ax[0], title="STORE 10 - LAWN AND GARDEN")
    train[(train.store_nbr == 36) & (train.family == "LADIESWEAR")].set_index("date").sales.plot(ax=ax[1], title="STORE 36 - LADIESWEAR")
    train[(train.store_nbr == 6) & (train.family == "SCHOOL AND OFFICE SUPPLIES")].set_index("date").sales.plot(ax=ax[2], title="STORE 6 - SCHOOL AND OFFICE SUPPLIES")
    train[(train.store_nbr == 14) & (train.family == "BABY CARE")].set_index("date").sales.plot(ax=ax[3], title="STORE 14 - BABY CARE")
    train[(train.store_nbr == 53) & (train.family == "BOOKS")].set_index("date").sales.plot(ax=ax[4], title="STORE 43 - BOOKS")
    st.pyplot(fig)

def fig_Train_d_family_patten(train):
    """
    Generate a graph to identify the daily sales pattern for each product family.

    Args:
        train (pandas DataFrame): The DataFrame containing the training data with the "date", "family", and "sales" columns.

    Returns:
        None

    Example:
        import pandas as pd
        import plotly.express as px
        import streamlit as st

        # Assume 'train' DataFrame is defined

        fig_Train_d_family_patten(train)
    """
    a = train.set_index("date").groupby("family").resample("D").sales.sum().reset_index()

    fig, ax = plt.subplots()
    fig = px.line(a, x="date", y="sales", color="family", title="Daily Total Sales of The Family")

    st.plotly_chart(fig)

def fig_Train_family_patten(train):
    """
    Generate a graph to identify the sales pattern for each product family.

    Args:
        train (pandas DataFrame): The DataFrame containing the training data with the "family" and "sales" columns.

    Returns:
        None

    Example:
        import pandas as pd
        import plotly.express as px
        import streamlit as st

        # Assume 'train' DataFrame is defined

        fig_Train_family_patten(train)
    """
    a = train.groupby("family").sales.mean().sort_values(ascending=False).reset_index()

    fig, ax = plt.subplots()
    fig = px.bar(a, y="family", x="sales", color="family", title="Which Product Family Preferred more?")

    st.plotly_chart(fig)

def fig_Train_Stores_patten(train, stores):
    """
    Generate a graph to identify the sales pattern for each store.

    Args:
        train (pandas DataFrame): The DataFrame containing the training data with the "store_nbr", "date", and "sales" columns.
        stores (pandas DataFrame): The DataFrame containing store information with the "store_nbr" and "city" columns.

    Returns:
        None

    Example:
        import pandas as pd
        import plotly.express as px
        import streamlit as st

        # Assume 'train' and 'stores' DataFrames are defined

        fig_Train_Stores_patten(train, stores)
    """
    d = pd.merge(train, stores)
    d["store_nbr"] = d["store_nbr"].astype("int8")
    d["year"] = d.date.dt.year

    fig, ax = plt.subplots()
    fig = px.line(d.groupby(["city", "year"]).sales.mean().reset_index(), x="year", y="sales", color="city")

    st.plotly_chart(fig)

def Feature_Engineering_Holidays(holidays, train, test, stores):
    """
    Perform preprocessing on holiday data.

    Args:
        holidays (pandas DataFrame): DataFrame containing holiday data with columns: "date", "type", "description", "transferred", "locale", "locale_name".
        train (pandas DataFrame): DataFrame containing training data with columns: "date", "store_nbr", "sales".
        test (pandas DataFrame): DataFrame containing test data with columns: "date", "store_nbr".
        stores (pandas DataFrame): DataFrame containing store information with columns: "store_nbr", "city", "state".

    Returns:
        pandas DataFrame: Preprocessed DataFrame with added holiday-related features.

    Example:
        import pandas as pd

        # Assume 'holidays', 'train', 'test', and 'stores' DataFrames are defined

        processed_data = Feature_Engineering_Holidays(holidays, train, test, stores)
    """
    ## Transferred Holidays(양도된 휴일) 처리
    tr1 = holidays[(holidays.type == "Holiday") & (holidays.transferred == True)].drop("transferred", axis=1).reset_index(drop=True)
    tr2 = holidays[(holidays.type == "Transfer")].drop("transferred", axis=1).reset_index(drop=True)
    tr = pd.concat([tr1, tr2], axis=1)
    tr = tr.iloc[:, [5, 1, 2, 3, 4]]

    holidays = holidays[(holidays.transferred == False) & (holidays.type != "Transfer")].drop("transferred", axis=1)
    holidays = pd.concat([holidays, tr]).reset_index(drop=True)

    ## Additional Holidays(추가된 휴일) 처리
    holidays["description"] = holidays["description"].str.replace("-", "").str.replace("+", "").str.replace("\d+","")
    holidays["type"] = np.where(holidays["type"] == "Additional", "Holiday", holidays["type"])

    ## Bridge Holidays(브릿지 휴일) 처리
    holidays["description"] = holidays["description"].str.replace("Puente ", "")
    holidays["type"] = np.where(holidays["type"] == "Bridge", "Holiday", holidays["type"])

    ## Work Day Holidays(근무 휴일(보상 휴일)) 처리
    work_day = holidays[holidays.type == "Work Day"]
    holidays = holidays[holidays.type != "Work Day"]

    ## Events are national(전국 행사) 처리
    events = holidays[holidays.type == "Event"].drop(["type", "locale", "locale_name"], axis=1).rename({"description": "events"}, axis=1)

    holidays = holidays[holidays.type != "Event"].drop("type", axis=1)
    regional = holidays[holidays.locale == "Regional"].rename({"locale_name": "state", "description": "holiday_regional"}, axis=1).drop("locale", axis=1).drop_duplicates()
    national = holidays[holidays.locale == "National"].rename({"description": "holiday_national"}, axis=1).drop(["locale", "locale_name"], axis=1).drop_duplicates()
    local = holidays[holidays.locale == "Local"].rename({"description": "holiday_local", "locale_name": "city"}, axis=1).drop("locale", axis=1).drop_duplicates()

    d = pd.merge(pd.concat([train, test]), stores)
    d["store_nbr"] = d["store_nbr"].astype("int8")
    ## National Holidays & Events(공휴일 및 이벤트)
    d = pd.merge(d, national, how="left")
    ## Regional(state 별)
    d = pd.merge(d, regional, how="left", on=["date", "state"])
    ## Local(city 별)
    d = pd.merge(d, local, how="left", on=["date", "city"])
    ## Work Day(실제 근무일 컬럼이 생성되면 제거)
    d = pd.merge(d, work_day[["date", "type"]].rename({"type": "IsWorkDay"}, axis=1), how="left")
    ## EVENT
    events["events"] = np.where(events.events.str.contains("futbol"), "Futbol", events.events)

    events, events_cat = one_hot_encoder(events, nan_as_category=False)
    events["events_Dia_de_la_Madre"] = np.where(events.date == "2016-05-08", 1, events["events_Dia_de_la_Madre"])
    events = events.drop(239)

    d = pd.merge(d, events, how="left")
    d[events_cat] = d[events_cat].fillna(0)

    ## NEW features
    d["holiday_national_binary"] = np.where(d.holiday_national.notnull(), 1, 0)
    d["holiday_local_binary"] = np.where(d.holiday_local.notnull(), 1, 0)
    d["holiday_regional_binary"] = np.where(d.holiday_regional.notnull(), 1, 0)

    d["national_independence"] = np.where(d.holiday_national.isin(["Batalla de Pichincha", "Independencia de Cuenca", "Independencia de Guayaquil", "Independecia de Guayaquil", "Primer Grito de Independencia"]), 1, 0)
    d["local_cantonizacio"] = np.where(d.holiday_local.str.contains("Cantonizacio"), 1, 0)
    d["local_fundacion"] = np.where(d.holiday_local.str.contains("Fundacion"), 1, 0)
    d["local_independencia"] = np.where(d.holiday_local.str.contains("Independencia"), 1, 0)

    holidays, holidays_cat = one_hot_encoder(d[["holiday_national", "holiday_regional", "holiday_local"]], nan_as_category=False)
    d = pd.concat([d.drop(["holiday_national", "holiday_regional", "holiday_local"], axis=1), holidays], axis=1)

    he_cols = d.columns[d.columns.str.startswith("events")].tolist() + d.columns[d.columns.str.startswith("holiday")].tolist() + d.columns[d.columns.str.startswith("national")].tolist() + \
              d.columns[d.columns.str.startswith("local")].tolist()
    d[he_cols] = d[he_cols].astype("int8")

    d[["family", "city", "state", "type"]] = d[["family", "city", "state", "type"]].astype("category")

    return d

def plot_stats(df, column, ax, color, angle):
    """
    Plot statistics for a given column.

    Args:
        df (pandas DataFrame): DataFrame containing the data.
        column (str): Name of the column to plot.
        ax (matplotlib Axes): Axes object to plot on.
        color (str): Color palette for the plot.
        angle (int): Rotation angle for x-axis labels.

    Returns:
        None

    Example:
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd

        # Assume 'df', 'column', 'ax', 'color', and 'angle' are defined

        plot_stats(df, column, ax, color, angle)
    """
    ## 선택된 열의 value값 계산
    count_classes = df[column].value_counts()
    ## 계산된 value를 인덱스를 행기준으로, count값을 열기준으로 barplot 생성
    ax = sns.barplot(x=count_classes.index, y=count_classes, ax=ax, palette=color)
    ax.set_title(column.upper(), fontsize=18)
    for tick in ax.get_xticklabels():
        tick.set_rotation(angle)

def grouped(df, key, freq, col):
    """
    Group data at a specific frequency.

    Args:
        df (pandas DataFrame): DataFrame containing the data.
        key (str): Column name to group by.
        freq (str): Frequency at which to group the data ('D', 'W', 'M', etc.).
        col (str): Column name to calculate the mean.

    Returns:
        pandas DataFrame: Grouped DataFrame with the mean value for the specified column.

    Example:
        import pandas as pd

        # Assume 'df', 'key', 'freq', and 'col' are defined

        df_grouped = grouped(df, key, freq, col)
    """
    df_grouped = df.groupby([pd.Grouper(key=key, freq=freq)]).agg(mean=(col, 'mean'))
    ## pd.Grouper = 그룹화의 기준이 될 시간 간격('freq')과 그룹화할 기준이 될 열('key')을 설정
    ## freq = 'D','W','M'과 같은 문자열로 나타냄
    df_grouped = df_grouped.reset_index()
    return df_grouped

def add_time(df, key, freq, col):
    """
    Add a 'time' column to the DataFrame.

    Args:
        df (pandas DataFrame): DataFrame to which the 'time' column will be added.
        key (str): Column name to group by.
        freq (str): Frequency at which to group the data ('D', 'W', 'M', etc.).
        col (str): Column name to calculate the mean.

    Returns:
        pandas DataFrame: DataFrame with the 'time' column added.

    Example:
        import pandas as pd

        # Assume 'df', 'key', 'freq', and 'col' are defined

        df_with_time = add_time(df, key, freq, col)
    """
    df_grouped = grouped(df, key, freq, col)
    df_grouped['time'] = np.arange(len(df_grouped.index))
    column_time = df_grouped.pop('time')
    df_grouped.insert(1, 'time', column_time)
    return df_grouped



def add_lag(df,key,freq,col,lag):
    """
    Add lag to the DataFrame.

    Args:
        df (pandas DataFrame): DataFrame to which the lag column will be added.
        key (str): Column name to group by.
        freq (str): Frequency at which to group the data ('D', 'W', 'M', etc.).
        col (str): Column name to calculate the mean.
        lag (int): Number of periods to lag the data.

    Returns:
        pandas DataFrame: DataFrame with the lag column added.

    Example:
        import pandas as pd

        # Assume 'df', 'key', 'freq', 'col', and 'lag' are defined

        df_with_lag = add_lag(df, key, freq, col, lag)
    """
    ## grouped 함수 호출
    df_grouped = grouped(df,key,freq,col)

    name = 'Lag_'+str(lag)
    ## 지연(lag)값 만큼 행을 밀어주며  df_groupd 데이터 프레임에 Lag열 생성
    df_grouped['Lag'] = df_grouped['mean'].shift(lag)
    return df_grouped

def W_M_Sales(train):
    """
    Plot weekly and monthly sales.

    Args:
        train (pandas DataFrame): DataFrame containing the sales data.

    Returns:
        None

    Example:
        import pandas as pd

        # Assume 'train' DataFrame is defined

        W_M_Sales(train)
    """
    df_grouped_train_w = add_time(train, 'date', 'W', 'sales')
    df_grouped_train_m = add_time(train, 'date', 'M', 'sales')


    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30, 8))

    # TRANSACTIONS (WEEKLY)
    axes[0].plot('time', 'mean', data=df_grouped_train_w, color='0.75')
    axes[0].set_title("Sales (grouped by week)", fontsize=20)
    # linear regression
    axes[0] = sns.regplot(x='time',
                          y='mean',
                          data=df_grouped_train_w,
                          scatter_kws=dict(color='0.75'),
                          ax=axes[0])

    # SALES (MONTHLY)
    axes[1].plot('time', 'mean', data=df_grouped_train_m, color='0.75')
    axes[1].set_title("Sales (grouped by month)", fontsize=20)
    # linear regression
    axes[1] = sns.regplot(x='time',
                          y='mean',
                          data=df_grouped_train_m,
                          scatter_kws=dict(color='0.75'),
                          line_kws={"color": "red"},
                          ax=axes[1])

    st.pyplot(fig)


def W_M_lag(train):
    """
    Plot lagged sales for weeks and months.

    Args:
        train (pandas DataFrame): DataFrame containing the sales data.

    Returns:
        None

    Example:
        import pandas as pd

        # Assume 'train' DataFrame is defined

        W_M_lag(train)
    """
    ## 주별 지연(lag)값 1 추가
    df_grouped_train_w_lag1 = add_lag(train, 'date', 'W', 'sales', 1)
    ## 월별 지연(lag)값 1 추가
    df_grouped_train_m_lag1 = add_lag(train, 'date', 'M', 'sales', 1)

    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(30,8))
    ax[0].plot('Lag','mean', data = df_grouped_train_w_lag1, color='0.75', linestyle=(0,(1,10)))
    ax[0].set_title('Sales (grouped by week)', fontsize=20)
    ax[0] = sns.regplot(x='Lag'
                        , y = 'mean'
                        , data = df_grouped_train_w_lag1
                        , scatter_kws=dict(color='0.75')
                        , ax=ax[0])

    ax[1].plot('Lag','mean', data = df_grouped_train_m_lag1, color='0.75', linestyle=(0, (1,10)))
    ax[1].set_title('Sales (grouped by month)', fontsize=20)
    ax[1] = sns.regplot(x='Lag'
                        , y = 'mean'
                        , data = df_grouped_train_m_lag1
                        , scatter_kws=dict(color='0.75')
                        , line_kws={'color':'red'}
                        , ax=ax[1])
    st.pyplot(fig)

def plot_moving_average(df,key,freq,col,window,min_periods,ax,title):
    """
    Plot moving average.

    Args:
        df (pandas DataFrame): DataFrame containing the data.
        key (str): Key column to group the data.
        freq (str): Frequency for grouping ('D', 'W', 'M', etc.).
        col (str): Column to calculate the moving average.
        window (int): Size of the moving average window.
        min_periods (int): Minimum number of periods required to calculate the moving average.
        ax (matplotlib Axes): Axes object to plot the data.
        title (str): Title of the plot.

    Returns:
        None

    Example:
        import pandas as pd
        import matplotlib.pyplot as plt

        # Assume 'df' DataFrame is defined
        fig, ax = plt.subplots()
        plot_moving_average(df, 'date', 'W', 'sales', 3, 2, ax, 'Moving Average Plot')
        plt.show()
    """
    df_grouped = grouped(df,key,freq,col)
    plot_moving_average = df_grouped['mean'].rolling(window=window, center = True, min_periods=min_periods).mean()
    # rolling = 이동평균 계산 함수
    # window = 몇 개씩 연산할지 입력    center = 중간을 기준으로 이동평균   min_period = 평균 낼 데이터의 최소 개수
    ax = df_grouped['mean'].plot(color = '0.75', linestyle = 'dashdot', ax=ax)
    ax = plot_moving_average.plot(linewidth=3, color='g', ax=ax)
    ax.set_title(title, fontsize=18)


def Trend_Moving_average(train):
    """
    Plot trend moving average graphs.

    Args:
        train (pandas DataFrame): DataFrame containing the training data.

    Returns:
        None

    Example:
        import pandas as pd
        import matplotlib.pyplot as plt

        # Assume 'train' DataFrame is defined
        Trend_Moving_average(train)
    """
    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(30,8))
    # plot_moving_average(transactions, 'date','W','transactions',7,4,ax[0],'Transactions Moving Average')
    plot_moving_average(train,'date','W','sales',7,4,ax[0],'Sales Weekly-7 Moving Average')
    plot_moving_average(train, 'date', 'W', 'sales', 30, 4, ax[1], 'Sales Weekly-30 Moving Average')

    st.pyplot(fig)


def eda_app():
    """
        Exploratory Data Analysis (EDA) application.

        Returns:
            None

        Example:
            eda_app()
    """
    train, test, transactions, stores, oil, holidays = utils.load_data()
    st.write('---')
    selected_data = st.radio('SELECT DATA', ["Train", "Transactions", "Oil"])
    st.write('---')
    st.write(' ')
    st.subheader(f"Exploratory Data Analysis(EDA) - {selected_data} DATA")

    # Datetime
    train["date"] = pd.to_datetime(train.date)
    test["date"] = pd.to_datetime(test.date)
    transactions["date"] = pd.to_datetime(transactions.date)
    oil["date"] = pd.to_datetime(oil.date)
    holidays["date"] = pd.to_datetime(holidays.date)

    # Data types
    train.onpromotion = train.onpromotion.astype("float16")
    train.sales = train.sales.astype("float32")
    stores.cluster = stores.cluster.astype("int8")

    # Train
    if selected_data == "Train":
        selected_chart = st.selectbox("Select Chart",
                                      ["Daily Total Sales of The Stores", "Daily Total Sales of The Family"
                                     , "Lags Grouped by Week / Month", "Moving Average", "Sales Grouped by Week / Month"])

        #        if selected_chart == "1":
        #            # 각 매장별 Sales 에 대한 상관 관계 그래프
        #            fig_Train_sales_Correlation(train)
        if selected_chart == "Daily Total Sales of The Stores":
            # 각 매장 별 Total Sales 패턴 파악
            fig_Train_store_TotalSales_patten(train)
            st.markdown('Train 데이터의 Sales 총합 데이터를 시계열 데이터로 표현했습니다. '
                        '21,22,29,42,52 총 5개의 매장에서 Sales가 0으로 나왔기에 정확한 데이터 분석을 위해 제거하였습니다.')
        ## 이상치 제거 : 매장별로 오픈하기 전의 시점
        train = train[~(train.store_nbr == 21)]
        train = train[~(train.store_nbr == 22)]
        train = train[~(train.store_nbr == 29)]
        train = train[~(train.store_nbr == 42)]
        train = train[~(train.store_nbr == 52)]

        ## 불필요한 값 제거 : 매장별로 판매하지 않는 제품 파악
        c = train.groupby(["store_nbr", "family"]).sales.sum().reset_index().sort_values(["family", "store_nbr"])
        c = c[c.sales == 0]

        outer_join = train.merge(c[c.sales == 0].drop("sales", axis=1), how="outer", indicator=True)
        train = outer_join[~(outer_join._merge == "both")].drop("_merge", axis=1)

        zero_prediction = []
        for i in range(0, len(c)):
            zero_prediction.append(pd.DataFrame({
                "date": pd.date_range("2017-08-16", "2017-08-31").tolist(),
                "store_nbr": c.store_nbr.iloc[i],
                "family": c.family.iloc[i],
                "sales": 0
            }))
        zero_prediction = pd.concat(zero_prediction)

        if selected_chart == "Daily Total Sales of The Family":
            ## 일별 제품 판매 패턴 파악
            col1, col2 = st.columns([4,4])
            with col1:
                fig_Train_d_family_patten(train)
            with col2:
                fig_Train_family_patten(train)

            st.markdown('제품군별 매충 총액 시계열 데이터입니다.'
                        '해당 차트를 통해 Grocery l, Beverage가 매출의 가장 큰 비중을 차지하는 것을 확인할 수 있엇습니다.')

        if selected_chart == "Lags Grouped by Week / Month":
            ## 지연(lag)에 따른 주,월별 판매액 평균
            W_M_lag(train)
            st.markdown(
                '머신러닝에서는 이전 데이터를 현재 데이터와 비교하면 좀 더 정확한 학습이 가능한 경우가 있습니다. '
                '이것을 lag(지연) 데이터라고 표현합니다. '
                'Sales grouped by month/week 데이터에 lag 데이터를 이용하여 나타내보았습니다.'
            )

        if selected_chart == "Sales Grouped by Week / Month":
            ## 판매액에 따른 주,월별 판매액 평균
            W_M_Sales(train)
            st.markdown(
                'train 데이터를 주별, 월별로 그룹화하여 각각 매출의 평균을 확인해보았고 '
                '선형회귀 분석을 통해 차트를 완성해보았습니다. '
                '회귀선의 양옆의 연한색 범위는 회귀분석 결과로부터 얻어진 회귀식이 실제로 모집단에서 존재하는 회귀식과 얼마나 유사한지를 나타내는 것입니다. 보통 이 신뢰 구간은 95%로 설정됩니다. '
                '따라서 이 그래프에서 회귀선 양옆의 영역은 실제 값이 예측된 회귀식에서 어느정도 벗어날 수 있는지를 나타내는 것입니다. '
                '만약 회귀선의 신뢰 구간이 매우 넓게 그려졌다면, 예측값의 신뢰도가 낮아진다는 것을 의미합니다.')

        if selected_chart == "Moving Average":
            ## 이동평균플롯으로 추세 파악
            Trend_Moving_average(train)
            st.markdown('주별 평균 판매액에 대한 시계열 데이터를 파악하기 위해 '
                        '7일 이동평균선과 7주 이동평균선을 사용하여 추세를 시각화 하였습니다.'
                        '이동평균은 이상치(Outlier)와 잡음(Noise)에 대한 문제점을 완화하는 특성을 가지고 있습니다.')


    # Transactions
    temp = pd.merge(train.groupby(["date", "store_nbr"]).sales.sum().reset_index(), transactions, how="left")

    if selected_data == "Transactions":
        selected_chart = st.selectbox("Select Chart",["Transactions Grouped by Stores", "Transactions Grouped by Month"
                                                    , "Correlation Transactions and Sales", "Transantioncs Grouped by Dayofweek"])

        if selected_chart == "Transactions Grouped by Stores":
            ## Transactions 과 Total Sales 간의 상관관계 패턴 파악
            fig_Transactions_TotalSales_Correlation(temp, transactions)
            st.markdown('Transactions은 안정적인 패턴을 가지고 있습니다. '
                        '2013년부터 2017년까지 박스 플롯에서 12월을 제외한 모든 월이 비슷합니다. '
                        '또한 이전 플롯에서 각 스토어에 대해 동일한 패턴을 보았습니다. '
                        '스토어 매출은 항상 연말에 증가했습니다.')

        if selected_chart == "Transactions Grouped by Month":
            ## Transactions 연도별, 월별 패턴 파악
            col1, col2 = st.columns([4,4])
            with col1:
                fig_Transactions_ym_patten1(transactions)
            with col2:
                fig_Transactions_ym_patten2(transactions)
            st.markdown('Transactions 데이터를 월별로 그룹화 하여 시각화 해본 결과'
                        '매년 연말(12월)의 Transactions가 급증하는 것을 확인할 수 있었습니다.')


        if selected_chart == "Correlation Transactions and Sales":
            ## Transactions 와 Sales 간의 상관관계 그래프
            fig_Transactions_Sales_Correlation(temp)
            st.markdown('총 매출과 거래량 간 그래프입니다.'
                        ' 그래프를 보면 빨강색 회귀선에 산점도가 가깝게 분포하는 것을 확인할 수 있으며, '
                        ' Sales와 Transactions 간의 높은 상관 관계가 있음을 알 수 있습니다.')


        if selected_chart == "Transantioncs Grouped by Dayofweek":
            ## Transactions 연도별, 요일별 패턴 파악
            fig_Transactions_ydw_patten(transactions)
            st.markdown('연도별 각 요일들의 평균 Transactions입니다.'
                        '주말에 높은 Transactions가 발생하는 것을 확인할 수 있습니다.')

    # Oil
    if selected_data == "Oil":
        selected_chart = st.selectbox("Select Chart", ["Daily Oil Price", "Correlation with Daily Oil Price", "Correlation with Family Oil Price"])

        if selected_chart == "Daily Oil Price":
            ## Oil Price 누락 값 추가
            fig_OilPrice(oil)
        if selected_chart == "Correlation with Daily Oil Price":
            ## Oil Price 와 Sales / Transactions 패턴 파악
            oil = oil.set_index("date").dcoilwtico.resample("D").sum().reset_index()
            oil["dcoilwtico"] = np.where(oil["dcoilwtico"] == 0, np.nan, oil["dcoilwtico"])
            oil["dcoilwtico_interpolated"] = oil.dcoilwtico.interpolate()
            fig_OilPrice_Sales_Transactions_patten(temp, oil)
        if selected_chart == "Correlation with Family Oil Price":
            ## Oil Price 와 제품군 별 Sales 패턴 파악
            oil = oil.set_index("date").dcoilwtico.resample("D").sum().reset_index()
            oil["dcoilwtico"] = np.where(oil["dcoilwtico"] == 0, np.nan, oil["dcoilwtico"])
            oil["dcoilwtico_interpolated"] = oil.dcoilwtico.interpolate()
            fig_OilPrice_family_patten(train, oil)
