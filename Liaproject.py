import streamlit as st

def append_menu()

def main():
  df_train = pd.read_csv('predict.csv')
  option = st.selectbox('what would you like to do', ('Append', 'Predict', 'Show results'))
  if option == "Append":
    append_menu()
  elif option == "Predict":
    predict_menu()
  elif option == "Show results":
    Result_menu()
    

if __name__ == "__main__":
    main()
