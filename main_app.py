# -*- coding: utf-8 -*-
import streamlit as st
from streamlit_option_menu import option_menu
import utils
import intro_app
import data_app
import eda_app
import stat_app
import ml_app


def main():
    """
        Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="Store Sales", page_icon=":💰:",
                            layout = "wide", initial_sidebar_state="expanded")
    # Streamlit 앱 실행
    with st.sidebar:
        selected = option_menu("Main Menu", ["INTRO", "DATA", "Exploratory Data Analysis", "STAT", "ML"],
                               icons=["house", "card-checklist", "bar-chart", "clipboard-data", "gear"],
                               menu_icon="cast",
                               default_index=0,
                               orientation="vertical")


    st.markdown("<h1 style='text-align: center;'> &#x1f4b0 Store Sales &#x1f4b0 </h1>", unsafe_allow_html=True)


    if selected == "INTRO":
        intro_app.intro_app()
    if selected == "DATA":
        data_app.data_app()
    if selected == "Exploratory Data Analysis":
        eda_app.eda_app()
    if selected == "STAT":
        stat_app.stat_app()
    if selected == "ML":
        ml_app.ml_app()

if __name__ == "__main__":
    main()