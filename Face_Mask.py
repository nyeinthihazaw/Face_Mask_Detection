# install joblib
# install opencv-python-headless

from pyexpat import model
import streamlit as st 
import opencv-python as cv2
import numpy as np 
import joblib 

model= joblib.load('https://github.com/nyeinthihazaw/Face_Mask_Detection/blob/main/models.dat')
st.header("Face MAsk Detection")
st.write("Choose any image :")
uploaded_file = st.file_uploader("Choose an image...")
if uploaded_file is not None:
	
	image = cv2.imread(uploaded_file)
	st.image(uploaded_file, caption='Input Image', use_column_width=True)
	img=cv2.resize(image,(240,240),interpolation=cv2.INTER_AREA)
	img_g=cv2.GaussianBlur(img,(3,3),0,0)
	img_m=cv2.medianBlur(img,3)
	img_c=cv2.Canny(img_g,50,150)
	fd=img_c.flatten()
	fd=fd.reshape(1,-1)
	result = model.predict(fd)
	if result == [1]:
		st.write('With Mask')
	elif result == [0]:
		st.write('Without Mask');
	else:
		st.write('No Face Here')
