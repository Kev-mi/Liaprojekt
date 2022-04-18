import streamlit as st
import pandas as pd
from sklearn import linear_model
from csv import writer
from datetime import datetime
import plotly.express as px
from scipy.stats import pearsonr


def duplicate_remover(df_duplicate):
    duplicate_bool = list(df_duplicate[["Length", "Height", "Width", "Price"]].duplicated())
    for x in range(0, len(duplicate_bool)):
        if duplicate_bool[x] == True:
            df_duplicate = df_duplicate.drop(x)


def csv_append(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)


def append_menu():
    with st.form("my_form"):
        st.write("Fan info")
        Width = st.text_input("Width")
        Height = st.text_input("Height")
        Length = st.text_input("Length")
        Price = st.text_input("Price")
        Date = st.text_input("Input date if not today (yyyy-mm-dd)")
        submitted = st.form_submit_button("Submit")
        if submitted:
            if Date == "":
                Date = datetime.date(datetime.now())
            row_contents = [Length, Height, Width, Price, Date]
            csv_append('Predict.csv', row_contents)


def predict_menu(df):
    with st.form("my_form"):
        st.write("Fan info")
        Width_pred = st.text_input("Width")
        Height_pred = st.text_input("Height")
        Length_pred = st.text_input("Length")
        submitted = st.form_submit_button("Predict price")
        if submitted:
            X = df[['Width', 'Height', 'Length']]
            y = df['Price']
            regr = linear_model.LinearRegression()
            regr.fit(X, y)
            predicted_price = regr.predict([[Width_pred, Height_pred, Length_pred]])
            st.write("price is " + str(predicted_price[0]) + "kr")


def result_menu(df):
    button = st.sidebar.button("Delete duplicates")
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
    df_train = pd.read_csv('Predict.csv')
    option = st.sidebar.selectbox('what would you like to do', ('Append', 'Predict', 'Show results', 'Show correlation'))
    if option == "Append":
        append_menu()
    elif option == "Predict":
        predict_menu(df_train)
    elif option == "Show results":
        result_menu(df_train)
    elif option == "Show correlation":
        correlation_menu(df_train)


if __name__ == "__main__":
    main()
