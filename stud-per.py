
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn.preprocessing import SplineTransformer,LabelEncoder

def load_model():
    with open("student_lr_final_model.pkl","rb") as file:
        model,scaler,le=pickle.load(file)
    return model,scaler,le

def preprocessing_input_data(data,scaler,le):
    data["Extracurricular_Activity"] = le.transform([data["Extracurricular_Activity"]])[0]
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

# this function to take data
def predict_data(data):
    model, scaler,le = load_model()
    processed_data = preprocessing_input_data(data,scaler,le)
    prediction = model.predict(processed_data)
    return prediction

def main():
    st.title("student performance prediction")
    st.write("enter your data to get a prediction for your performance")

    hour_studies = st.number_input("Hours studies",min_value = 1, max_value = 10, value = 5)
    previous_score = st.number_input("previous score",min_value = 40, max_value = 100, value = 70)
    extra = st.selectbox("extra curri activity", ["Yes", "No"])
    sleeping_hour = st.number_input("sleeping hours", min_value =4, max_value = 10, value = 7)
    number_of_paper_solved = st.number_input("number of question paper solved", min_value =0, max_value = 10, value = 5)

    if st.button("predict-your_score"):
        user_data = {
            "Hours_Studied":hour_studies,
            "Previous_Score":previous_score,
            "Extracurricular_Activity":extra,
            "Sleep_Hours":sleeping_hour,
            "Sample_QP_Practiced":number_of_paper_solved
        }
        prediction = predict_data(user_data)
        st.success(f"Your prediction result is {prediction}")

if __name__ == "__main__":
    main()

