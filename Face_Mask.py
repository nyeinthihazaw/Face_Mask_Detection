# install joblib
# install opencv-python-headless

from pyexpat import model
import streamlit as st 
import cv2
import numpy as np 
import joblib 
# import pillow as PIL
from PIL import Image

model= joblib.load('models.dat')
st.header("Face Mask Detection")
st.write("Choose any image :")
uploaded_file = st.file_uploader("Choose an image...")
if uploaded_file is not None:
	
	image = Image.open(uploaded_file)
	st.image(image, caption='Input', use_column_width=True)
	img_array = np.array(image)
	cv2.imwrite('out.png', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
	image =cv2.imread('out.png')
	img=cv2.resize(image,(240,240),interpolation=cv2.INTER_AREA)
	img_g=cv2.GaussianBlur(img,(3,3),0,0)
	img_m=cv2.medianBlur(img,3)
	img_c=cv2.Canny(img_g,50,150)
	st.image(img_c, caption='Input', use_column_width=True)
	fd=img_c.flatten()
	fd=fd.reshape(1,-1)
	result = model.predict(fd)
	if result == [1]:
		st.write('With Mask')
	elif result == [0]:
		st.write('Without Mask');
	else:
		st.write('No Face Here')
