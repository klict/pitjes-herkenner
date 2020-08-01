import numpy as np
import cv2
from algoritme import run_algoritme

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_CONVERT_RGB, True)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if frame.any():
        try:
            result = run_algoritme(frame)

            cv2.imshow('frame', result)
        except:
            cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
