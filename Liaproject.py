import streamlit as st
import pandas as pd
from sklearn import linear_model


def append_menu():
    with st.form("my_form"):
        st.write("Fan info")
        Width = st.text_input("Width")
        Height = st.text_input("Height")
        Length = st.text_input("Length")
        Price = st.text_input("Price")
        Date = st.text_input("Date")
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write("slider", slider_val, "checkbox", checkbox_val)


def predict_menu():
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
            st.write(predicted_price)


def result_menu():
    print("")


def correlation_menu():
    print("")


def main():
    df_train = pd.read_csv('Predict.csv')
    option = st.sidebar.selectbox('what would you like to do',
                                  ('Append', 'Predict', 'Show results', 'Show correlation'))
    if option == "Append":
        append_menu()
    elif option == "Predict":
        predict_menu()
    elif option == "Show results":
        result_menu()
    elif option == "Show correlation":
        correlation_menu()


if __name__ == "__main__":
    main()
