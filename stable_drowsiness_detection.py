import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

# ==========================================
# LOAD MODEL
# ==========================================

model = load_model("models/drowsiness_modelretrained.keras")

# ==========================================
# LOAD FACE CASCADE
# ==========================================

face_cascade = cv2.CascadeClassifier(
    'haarcascades/haarcascade_frontalface_default.xml'
)

# ==========================================
# START WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ==========================================
# CLAHE
# ==========================================

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8,8)
)

# ==========================================
# TEMPORAL SMOOTHING
# ==========================================

prediction_buffer = deque(maxlen=10)

closed_counter = 0
alert_active = False

# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Flip webcam
    frame = cv2.flip(frame, 1)

    # ==========================================
    # PREPROCESS FRAME
    # ==========================================

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    enhanced_gray = clahe.apply(gray)

    # ==========================================
    # FACE DETECTION
    # ==========================================

    faces = face_cascade.detectMultiScale(
        enhanced_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(120,120)
    )

    # ==========================================
    # PROCESS FACES
    # ==========================================

    for (x, y, w, h) in faces:

        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            (255,0,0),
            2
        )

        # ==========================================
        # GEOMETRIC EYE REGION EXTRACTION
        # ==========================================

        # Left eye region
        left_eye = frame[
            y + int(h * 0.22): y + int(h * 0.45),
            x + int(w * 0.12): x + int(w * 0.42)
        ]

        # Right eye region
        right_eye = frame[
            y + int(h * 0.22): y + int(h * 0.45),
            x + int(w * 0.58): x + int(w * 0.88)
        ]

        # Skip invalid crops
        if left_eye.size == 0 or right_eye.size == 0:
            continue

        # ==========================================
        # COMBINE EYES
        # ==========================================

        combined_eyes = np.hstack((left_eye, right_eye))

        # ==========================================
        # PREPROCESS FOR MODEL
        # ==========================================

        eye_input = cv2.resize(
            combined_eyes,
            (224,224)
        )

        eye_input = eye_input / 255.0

        eye_input = np.expand_dims(
            eye_input,
            axis=0
        )

        # ==========================================
        # PREDICTION
        # ==========================================

        prediction = model.predict(
            eye_input,
            verbose=0
        )[0][0]

        # ==========================================
        # TEMPORAL SMOOTHING
        # ==========================================

        prediction_buffer.append(prediction)

        avg_prediction = np.mean(prediction_buffer)

        # ==========================================
        # CLASSIFICATION
        # ==========================================

        if avg_prediction > 0.50:

            label = "OPEN EYES"
            color = (0,255,0)

            closed_counter = 0
            alert_active = False

        else:

            label = "CLOSED EYES"
            color = (0,0,255)

            closed_counter += 1

        # ==========================================
        # DROWSINESS ALERT
        # ==========================================

        if closed_counter > 15:

            alert_active = True

        # ==========================================
        # SHOW ALERT
        # ==========================================

        if alert_active:

            cv2.putText(
                frame,
                "DROWSINESS ALERT!",
                (50,100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                3
            )

        # ==========================================
        # DRAW EYE REGIONS
        # ==========================================

        cv2.rectangle(
            frame,
            (x + int(w * 0.12), y + int(h * 0.22)),
            (x + int(w * 0.42), y + int(h * 0.45)),
            color,
            2
        )

        cv2.rectangle(
            frame,
            (x + int(w * 0.58), y + int(h * 0.22)),
            (x + int(w * 0.88), y + int(h * 0.45)),
            color,
            2
        )

        # ==========================================
        # SHOW LABEL
        # ==========================================

        cv2.putText(
            frame,
            label,
            (x, y-15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        # ==========================================
        # SHOW CONFIDENCE
        # ==========================================

        cv2.putText(
            frame,
            f"Confidence: {avg_prediction:.2f}",
            (x, y+h+25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    # ==========================================
    # DISPLAY WINDOW
    # ==========================================

    cv2.imshow(
        "Final Driver Drowsiness Detection",
        frame
    )

    # ESC TO EXIT
    key = cv2.waitKey(1)

    if key == 27:
        break

# ==========================================
# CLEANUP
# ==========================================

cap.release()
cv2.destroyAllWindows()