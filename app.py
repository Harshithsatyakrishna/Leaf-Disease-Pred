import streamlit as st
import tensorflow as tf

import numpy as np
from PIL import Image # Strreamlit works with PIL library very easily for Images
import cv2

model_path="C:\\Users\\Harsha\\Downloads\\mymodel.h5"

st.title("Plant disease prediction ")
upload = st.file_uploader('Upload a plant image')


if upload is not None:
  file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
  opencv_image = cv2.imdecode(file_bytes, 1)
  opencv_image = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2RGB) # Color from BGR to RGB
  img = Image.open(upload)
  st.image(img,caption='Uploaded Image',width=300)
  if(st.button('Predict')):
    model = tf.keras.models.load_model(model_path)
    x = cv2.resize(opencv_image, dsize=(256,256))
    ar = np.array([x])
    y = model.predict(ar)
    ans=np.argmax(y,axis=1)
    if(ans==0):
      st.title('Apple___Apple_scab')
    if(ans==1):
      st.title('Apple___Black_rot')
    if(ans==2):
      st.title('Apple___Cedar_apple_rust')
    if(ans==3):
      st.title('Apple___healthy')
    if(ans==4):
      st.title('Blueberry___healthy')
    if(ans==5):
      st.title('Cherry_(including_sour)___Powdery_mildew')
    if(ans==6):
      st.title('Cherry_(including_sour)___healthy')
    if(ans==7):
      st.title('Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot')
    if(ans==8):
      st.title('Corn_(maize)___Common_rust_')
    if(ans==9):
      st.title('Corn_(maize)___Northern_Leaf_Blight')
    if(ans==10):
      st.title('Corn_(maize)___healthy')
    if(ans==11):
      st.title('Grape___Black_rot')
    if(ans==12):
      st.title('Grape___Esca_(Black_Measles)')
    if(ans==13):
      st.title('Grape___Leaf_blight_(Isariopsis_Leaf_Spot)')
    if(ans==14):
      st.title('Grape___healthy')
    if(ans==15):
      st.title('Orange___Haunglongbing_(Citrus_greening)')
    if(ans==16):
      st.title('Peach___Bacterial_spot')
    if(ans==17):
      st.title('Peach___healthy')
    if(ans==18):
      st.title('Pepper,_bell___Bacterial_spot')
    if(ans==19):
      st.title('Pepper,_bell___healthy')
    if(ans==20):
      st.title('Potato___Early_blight')
    if(ans==21):
      st.title('Potato___Late_blight')
    if(ans==22):
      st.title('Potato___healthy')
    if(ans==23):
      st.title('Raspberry___healthy')
    if(ans==24):
      st.title('Soybean___healthy')
    if(ans==25):
      st.title('Squash___Powdery_mildew')
    if(ans==26):
      st.title('Strawberry___Leaf_scorch')
    if(ans==27):
      st.title('Strawberry___healthy')
    if(ans==28):
      st.title('Tomato___Bacterial_spot')
    if(ans==29):
      st.title('Tomato___Early_blight')
    if(ans==30):
      st.title('Tomato___Late_blight')
    if(ans==31):
      st.title('Tomato___Leaf_Mold')
    if(ans==32):
      st.title('Tomato___Septoria_leaf_spot')
    if(ans==33):
      st.title('Tomato___Spider_mites Two-spotted_spider_mite')
    if(ans==34):
      st.title('Tomato___Target_Spot')
    if(ans==35):
      st.title('Tomato___Tomato_Yellow_Leaf_Curl_Virus')
    if(ans==36):
      st.title('Tomato___Tomato_mosaic_virus')
    if(ans==37):
      st.title('Tomato___healthy')

    


      