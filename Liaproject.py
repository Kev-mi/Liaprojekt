import streamlit as st
import pandas as pd
from sklearn import linear_model
from csv import writer
from datetime import datetime
import plotly.express as px


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


def result_menu():
    print("")


def correlation_menu(df):
    plot = px.scatter(data_frame=df, x=df["Length"], y="Price", trendline="ols")
    st.plotly_chart(plot)


def main():
    df_train = pd.read_csv('Predict.csv')
    option = st.sidebar.selectbox('what would you like to do', ('Append', 'Predict', 'Show results', 'Show correlation'))
    if option == "Append":
        append_menu()
    elif option == "Predict":
        predict_menu(df_train)
    elif option == "Show results":
        result_menu()
    elif option == "Show correlation":
        correlation_menu(df_train)


if __name__ == "__main__":
    main()
