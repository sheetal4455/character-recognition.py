import streamlit as st
import pandas as pd
import numpy as np
import kagglehub
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from streamlit_drawable_canvas import st_canvas

st.title("Character Recognition using Multilayer Perceptron (MLP)")

@st.cache_data
def load_dataset():
    path = kagglehub.dataset_download("sachinpatel21/az-handwritten-alphabets-in-csv-format")

    for file in os.listdir(path):
        if file.endswith(".csv"):
            csv_path = os.path.join(path, file)
            df = pd.read_csv(csv_path)
            df = df.sample(70000, random_state=42)
            return df

st.write("Loading dataset...")
df = load_dataset()
st.success("Dataset Loaded")

X = df.iloc[:,1:].values
y = df.iloc[:,0].values

X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

@st.cache_resource
def train_model(X_train, y_train):
    model = MLPClassifier(
        hidden_layer_sizes=(128,64),
        max_iter=30,
        learning_rate_init=0.001,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

with st.spinner("Training Model..."):
    mlp = train_model(X_train, y_train)

st.success("Model Training Completed")

y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write(f"Accuracy : {round(accuracy*100,2)} %")

st.subheader("Draw a Character")

canvas_result = st_canvas(
    stroke_width=22,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

def preprocess_image(img):
    img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(img,50,255,cv2.THRESH_BINARY)

    contours,_ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        c = max(contours,key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        img = img[y:y+h,x:x+w]

    h,w = img.shape

    if h>w:
        new_h = 20
        new_w = int(w*(20/h))
    else:
        new_w = 20
        new_h = int(h*(20/w))

    img = cv2.resize(img,(new_w,new_h))

    canvas = np.zeros((28,28))

    x_offset = (28-new_w)//2
    y_offset = (28-new_h)//2

    canvas[y_offset:y_offset+new_h,x_offset:x_offset+new_w] = img

    img = 255 - canvas
    img = img/255.0

    return img.reshape(1,784)

if canvas_result.image_data is not None:
    processed = preprocess_image(canvas_result.image_data)

    prediction = mlp.predict(processed)

    letter = chr(prediction[0] + 65)

    st.subheader("Predicted Character")
    st.success(letter)
