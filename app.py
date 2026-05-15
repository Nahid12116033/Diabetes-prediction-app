
import gradio as gr
import pickle
import numpy as np

with open("diabetes_model.pkl", "rb") as file:
    model = pickle.load(file)

def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]],dtype=float)
  
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        return "the person is predicted to have diabetes."
    else:
        return "the person is predicted to not have diabetes."



interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose"),
        gr.Number(label="BloodPressure"),
        gr.Number(label="SkinThickness"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="DiabetesPedigreeFunction"),
        gr.Number(label="Age")
        ],

        outputs=gr.Textbox(label="prediction"),
        title="diabetes prediction",

        description="enter the patient's information to predict diabetes."
)

interface.launch()
