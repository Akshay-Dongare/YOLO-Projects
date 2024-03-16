import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
import math

model = YOLO("./runs/detect/train2/weights/best.pt")

# object classes
classNames = ['birthday','id','name']


st.set_page_config(
  page_title="ID Card Detection",
  page_icon="🤖",
  layout="wide",
  initial_sidebar_state="expanded"
)

st.title("ID Card Detection")

st.markdown("This app for ID Card Detection using YOLOv9 model")

img_files = st.file_uploader("Choose image file to detect", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

for n, img_file_buffer in enumerate(img_files):
  if img_file_buffer is not None:
    # we'll do this later
    # 1) image file buffer will converted to cv2 image
    # function to convert file buffer to cv2 image
    def create_opencv_image_from_stringio(img_stream, cv2_img_flag=1):
        img_stream.seek(0)
        img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2_img_flag)

    # 1) image file buffer will converted to cv2 image
    open_cv_image = create_opencv_image_from_stringio(img_file_buffer)
    # 2) pass image to the model to get the detection result

    # 3) show result image using st.image()
    if open_cv_image is not None:
        st.image(open_cv_image, channels="BGR", caption=f'Detection Results ({n+1}/{len(img_files)})')


img_file_buffer  = st.camera_input("Take a webcam pic of id card")
if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    results = model(cv2_img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(cv2_img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(cv2_img, classNames[cls], org, font, fontScale, color, thickness)

    st.image(cv2_img)



st.markdown("""
  <p style='text-align: center; font-size:16px; margin-top: 32px'>
    Akshay Dongare @2024
  </p>
""", unsafe_allow_html=True)

