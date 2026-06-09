import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import os
import time


# ==================================================
# PAGE
# ==================================================

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    layout="wide"
)


# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        os.path.join(
            "models",
            "drowsiness_modelretrained.keras"
        )
    )


model = load_model()


# ==================================================
# LOAD FACE CASCADE
# ==================================================

FACE_PATH = os.path.join(

    "haarcascades",

    "haarcascade_frontalface_default.xml"

)

face_cascade = cv2.CascadeClassifier(
    FACE_PATH
)


# ==================================================
# SESSION STATE
# ==================================================

if "running" not in st.session_state:

    st.session_state.running = False


def toggle():

    st.session_state.running = (
        not st.session_state.running
    )


# ==================================================
# PREDICT
# ==================================================

def predict_eye(img):

    img = cv2.resize(
        img,
        (224,224)
    )

    img = img.astype(
        np.float32
    )

    img /= 255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    return model.predict(
        img,
        verbose=0
    )[0][0]


# ==================================================
# HEADER
# ==================================================

st.markdown(
"""
<div style='text-align:center;'>

# Driver Drowsiness Detection

Real-time drowsiness detection using Deep Learning,
face localisation and geometric eye extraction.

This system analyses facial regions from webcam input,
extracts both eye regions,
and classifies eye state to trigger alerts.

</div>
""",

unsafe_allow_html=True
)


# ==================================================
# CAMERA
# ==================================================

left_cam, center_cam, right_cam = st.columns(
    [1.5, 4, 1.5]
)

with center_cam:

    frame_placeholder = st.empty()

left, center, right = st.columns(
    [3,1,3]
)

left_btn, center_btn, right_btn = st.columns(
    [4,1,4]
)

with center_btn:

    

    if st.session_state.running:

        st.button(
            "⏹ Stop Webcam",
            on_click=toggle
        )

    else:

        st.button(
            "▶ Start Webcam",
            on_click=toggle
        )


# ==================================================
# SETTINGS
# ==================================================

st.divider()

c1, c2 = st.columns(2)

with c1:

    threshold = st.slider(

        "Classification Threshold",

        0.1,

        0.9,

        0.50
    )


with c2:

    alert_time = st.slider(

        "Alert Time (seconds)",

        1,

        5,

        2
    )


# ==================================================
# WEBCAM LOOP
# ==================================================

if st.session_state.running:


    cap = cv2.VideoCapture(0)

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )


    clahe = cv2.createCLAHE(

        clipLimit=2,

        tileGridSize=(8,8)
    )


    prediction_buffer = deque(
        maxlen=10
    )

    closed_start = None


    while (

        cap.isOpened()

        and

        st.session_state.running

    ):


        ret, frame = cap.read()

        if not ret:

            break


        frame = cv2.flip(
            frame,
            1
        )


        gray = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2GRAY
        )


        enhanced = clahe.apply(
            gray
        )


        faces = face_cascade.detectMultiScale(

            enhanced,

            scaleFactor=1.1,

            minNeighbors=5,

            minSize=(120,120)
        )


        for (

            x,
            y,
            w,
            h

        ) in faces:


            cv2.rectangle(

                frame,

                (x,y),

                (x+w,y+h),

                (255,0,0),

                3
            )


            left_eye = frame[

                y+int(h*0.22):
                y+int(h*0.45),

                x+int(w*0.12):
                x+int(w*0.42)
            ]


            right_eye = frame[

                y+int(h*0.22):
                y+int(h*0.45),

                x+int(w*0.58):
                x+int(w*0.88)
            ]


            if (

                left_eye.size==0

                or

                right_eye.size==0

            ):

                continue


            combined = np.hstack(

                (

                    left_eye,

                    right_eye
                )
            )


            pred = predict_eye(
                combined
            )


            prediction_buffer.append(
                pred
            )


            smooth = np.mean(
                prediction_buffer
            )


            if smooth > threshold:

                state = "OPEN"

                color = (
                    0,
                    255,
                    0
                )

                closed_start = None


            else:

                state = "CLOSED"

                color = (
                    0,
                    0,
                    255
                )

                if closed_start is None:

                    closed_start = time.time()


            alert_active = False


            if closed_start:

                elapsed = (

                    time.time()

                    -

                    closed_start
                )

                if elapsed >= alert_time:

                    alert_active = True


            if alert_active:

                cv2.rectangle(

                    frame,

                    (142,60),

                    (510,140),

                    (0,0,255),

                    -1
                )


                cv2.putText(

                    frame,

                    "DROWSINESS ALERT",

                    (145,110),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    1.2,

                    (

                        255,

                        255,

                        255

                    ),

                    3
                )


            # Left Eye

            cv2.rectangle(

                frame,

                (

                    x+int(w*0.12),

                    y+int(h*0.22)

                ),

                (

                    x+int(w*0.42),

                    y+int(h*0.45)

                ),

                color,

                2
            )


            # Right Eye

            cv2.rectangle(

                frame,

                (

                    x+int(w*0.58),

                    y+int(h*0.22)

                ),

                (

                    x+int(w*0.88),

                    y+int(h*0.45)

                ),

                color,

                2
            )


            cv2.putText(

                frame,

                f"{state}",

                (

                    x,

                    y-10

                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                color,

                2
            )


            cv2.putText(

                frame,

                f"Confidence: {smooth:.2f}",

                (

                    x,

                    y+h+25

                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                color,

                2
            )


        frame_placeholder.image(

            cv2.cvtColor(

                frame,

                cv2.COLOR_BGR2RGB
            ),

            use_container_width=True
        )


    cap.release()