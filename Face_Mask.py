from pyexpat import model
import streamlit as st 
import cv2
import numpy as np 
import joblib 
from PIL import Image

model= joblib.load('models.dat')
st.header("Face Mask Detection")
st.write("Choose any image :")
uploaded_file = st.file_uploader("Choose an image...")
min = st.slider('Minimun Intensity for Canny Edge', 0, 255, 255)
max = st.slider('Maximun Intensity for Canny Edge', 0, 255, 255)
# kernel_size = st.selectbox('Kernel size for Gaussian Blur',(3, 5)
if uploaded_file is not None:
	
	image = Image.open(uploaded_file)
	st.image(image, caption='Input', width=300)
	img_array = np.array(image)
	cv2.imwrite('out.png', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
	image =cv2.imread('out.png')
	img=cv2.resize(image,(240,240),interpolation=cv2.INTER_AREA)
	img_g=cv2.GaussianBlur(img,(3,3),0,0)
# 	img_m=cv2.medianBlur(img,3)
	img_c=cv2.Canny(img_g,min,max)
# 	st.image(img_g, caption='Gaussian Blur', use_column_width=True)
	st.image(img_c, caption='Canny Edge Detection', width=300)
	fd=img_c.flatten()
	fd=fd.reshape(1,-1)
	result = model.predict(fd)
	if result == [1]:
		st.markdown(f'<h1 style="color:green;font-size:24px;">{"With Mask"}</h1>', unsafe_allow_html=True)
# 		st.write('With Mask')
	elif result == [0]:
		st.markdown(f'<h1 style="color:red;font-size:24px;">{"Without Mask"}</h1>', unsafe_allow_html=True)
# 		st.write('Without Mask')
	else:
		st.markdown(f'<h1 style="color:white;font-size:24px;">{"No Face Here"}</h1>', unsafe_allow_html=True)
# 		st.write('No Face Here')

