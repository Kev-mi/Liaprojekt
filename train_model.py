import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('dimensions.csv')

X = df[["Length", "Height", "Width"]]
y = df["Price"]

regr = linear_model.LinearRegression()
regr.fit(X,y)

def price_predictor(Length_par, Height_par, Width_par):
    Predicted_price_function = regr.predict([[Length_par, Height_par, Width_par]])
    return Predicted_price_function


Length,Height, Width = input("Enter Length, Height and Width in that order to predict price")
Predicted_price = Price_predictor(Length,Height, Width)
