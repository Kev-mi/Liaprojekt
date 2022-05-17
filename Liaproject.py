import streamlit as st
import pandas as pd
from sklearn import linear_model
from csv import writer
from datetime import datetime
import plotly.express as px
from scipy.stats import pearsonr
import datetime as dt
import numpy as np
import math
from geopy.distance import geodesic


def fan_number_calc(df_for_calc,room_length, room_width):
    fans_length = [room_length]*len(df_for_calc)
    fans_width = [room_width]*len(df_for_calc)
    df_for_calc.insert(2, 'Number of fans along width', fans_length, True)
    df_for_calc.insert(2, 'Number of fans along along length', fans_width, True)
    df_for_calc['Number of fans along width'] = df_for_calc['Number of fans along width'].div(df_for_calc['Fan Diameter (coverage)'].values,axis=0).div(float(0.63829)).apply(np.floor)
    df_for_calc['Number of fans along along length'] = df_for_calc['Number of fans along along length'].div(df_for_calc['Fan Diameter (coverage)'].values,axis=0).div(float(0.63829)).apply(np.floor)
    st.write(df_for_calc)


def df_filter_function(df_f, Length_filter, Height_filter, Width_filter, Price_filter, year_filter, year_filter2):
    b = df_f[df_f['Length'].between(Length_filter[0], Length_filter[1])]
    c = df_f[df_f['Height'].between(Height_filter[0], Height_filter[1])]
    d = df_f[df_f['Width'].between(Width_filter[0], Width_filter[1])]
    e = df_f[df_f['Price'].between(Price_filter[0], Price_filter[1])]
    start_date = year_filter + "-01-01"
    end_date = year_filter2 + "-12-31"
    mask = (df_f["Date"] >= start_date) & (df_f["Date"] <= end_date)
    a = df_f[mask]
    return a[(a.isin(b)) & (a.isin(c)) & (a.isin(d)) & (a.isin(e))].dropna()


def duplicate_remover(df_duplicate):
    duplicate_bool = list(df_duplicate[["Length", "Height", "Width", "Price"]].duplicated())
    for x in range(0, len(duplicate_bool)):
        if duplicate_bool[x] == True:
            df_duplicate = df_duplicate.drop(x)
    df_duplicate.to_csv('Train.csv', index=False)


def csv_append(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow("")
        csv_writer.writerow(list_of_elem)
        df_test = pd.read_csv('Train.csv')
        return df_test
    
    
def convert_df_to_csv(df_pd_to_csv):
  return df_pd_to_csv.to_csv().encode('utf-8')


def append_menu(df_unconverted):
    with st.form("my_form"):
        st.write("Building info")
        Width = st.text_input("Building Width (meter)")
        Height = st.text_input("Building Height (meter)")
        Length = st.text_input("Building Length (meter)")
        Price = st.text_input("Price (tusen sek)")
        Date = st.text_input("Input date if not today (yyyy-mm-dd)")
        submitted = st.form_submit_button("Submit")
        if submitted:
            if Date == "":
                Date = datetime.date(datetime.now())
            row_contents = [Length, Height, Width, Price, Date]
            local_csv = csv_append('Train.csv', row_contents)
    try:
        local_csv = convert_df_to_csv(df_unconverted)
        st.download_button('Download CSV', local_csv, 'train.csv')
        with open('myfile.csv') as f:
            st.download_button('Download CSV', f)
    except FileNotFoundError:
        pass


def predict_menu(df):
    with st.form("my_form"):
        st.write("Building info")
        train_year = st.sidebar.selectbox("Select year to use data from", sorted(set(pd.DatetimeIndex(df['Date']).year)))
        train_month = st.sidebar.selectbox("Select month to use data from",sorted(set(pd.DatetimeIndex(df['Date']).month)))
        Width_pred = st.text_input("Building Width (meter)")
        city_1 = st.sidebar.selectbox("Select which city to start from", sorted("Malmö göteborg stockolm"))
        city_2 = st.sidebar.selectbox("Select which city to start from", sorted("Malmö göteborg stockolm"))
        Height_pred = st.text_input("Building Height (meter)")
        Length_pred = st.text_input("Building Length (meter)")
        submitted = st.form_submit_button("Predict price")
        font_size = st.sidebar.slider("Enter text size", 10, 100, value=20)
        if submitted:
            testing = pd.to_datetime((str(train_year) + "-" + str(train_month) + "-" + "01")).date()
            for x in range(0, len((df['Date']))):
                df['Date'][x] = pd.to_datetime(df['Date'][x])
            df_pred = df.loc[((df['Date'] > testing)).dropna()]
            X = df_pred[['Width', 'Height', 'Length']]
            y = df_pred['Price']
            regr = linear_model.LinearRegression()
            regr.fit(X, y)
            predicted_price = regr.predict([[Width_pred, Height_pred, Length_pred]])
            df_fans = fan_number_calc(pd.read_csv('fans.csv'), float(Length_pred), float(Width_pred))
            string_output = "price is " + str(math.floor(predicted_price[0])) + "tkr" + " (exklusive resekostnader)" + "fr.o.m. " + str(train_year) + "-" + str(train_month) +"-dd"
            html_str = f"""<style>p.a{{font:bold {font_size}px Courier;}}</style><p class="a">{string_output}</p> """
            st.markdown(html_str, unsafe_allow_html=True)
            string_output_2 = "If used in a heated building:"
            string_output_3 = str(round(1.4*int(Height_pred),2))+ " Temperaturskillnad mellan golv och tak"
            string_output_4 = str(int(Height_pred)*3)+ "% besparing"
            html_str_2 = f"""<style>p.a{{font:bold {font_size}px Courier;}}</style><p class="a">{string_output_2}</p> """
            html_str_3 = f"""<style>p.a{{font:bold {font_size}px Courier;}}</style><p class="a">{string_output_3}</p> """
            html_str_4 = f"""<style>p.a{{font:bold {font_size}px Courier;}}</style><p class="a">{string_output_4}</p> """
            st.markdown(html_str_2, unsafe_allow_html=True)
            st.markdown(html_str_3, unsafe_allow_html=True)
            st.markdown(html_str_4, unsafe_allow_html=True)



def result_menu(df):
    year_list = []
    button = st.sidebar.button("Delete duplicates")
    Length_slider = st.sidebar.slider("Select length range", value=[int(df["Length"].min()), int(df["Length"].max())])
    Height_slider = st.sidebar.slider("Select height range", value=[int(df["Height"].min()), int(df["Height"].max())])
    Width_slider = st.sidebar.slider("Select width range", value=[int(df["Width"].min()), int(df["Width"].max())])
    Price_slider = st.sidebar.slider("Select price range", value=[int(df["Price"].min()), int(df["Price"].max())])
    for x in df.index:
        year_list.append(str(df["Date"][x])[:4])
    year_list = set(year_list)
    selected_year = st.sidebar.selectbox("Select year lower bound", sorted(year_list))
    selected_year_2 = st.sidebar.selectbox("Select year upper bound", sorted(year_list), index=len(set(year_list))-1)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df = df_filter_function(df, Length_slider, Height_slider, Width_slider, Price_slider, selected_year, selected_year_2)
    df['Date'] = pd.to_datetime(df['Date'])
    df["Date"] = df["Date"].dt.date
    st.write(df)
    if button:
        duplicate_remover(df)


def correlation_menu(df):
    x_var = st.sidebar.selectbox('select what you want to see correlation with price', ('Length', 'Height', 'Width'))
    plot = px.scatter(data_frame=df, x=df[x_var], y="Price", trendline="ols")
    corr, _ = pearsonr(df[x_var], df["Price"])
    st.write(str(corr)[0:5])
    st.plotly_chart(plot)


def main():
    df_train = pd.read_csv('Train.csv')
    option = st.sidebar.selectbox('what would you like to do', ('Append', 'Predict', 'Show data', 'Show correlation'))
    if option == "Append":
        append_menu(df_train)
    elif option == "Predict":
        predict_menu(df_train)
    elif option == "Show data":
        result_menu(df_train)
    elif option == "Show correlation":
        correlation_menu(df_train)


if __name__ == "__main__":
    main()
