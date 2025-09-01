from functools import lru_cache
from typing import Tuple, List
import cv2
import numpy as np


# Tải các bộ phân loại
@lru_cache(maxsize=1)
def _load_classifiers():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    if face_cascade.empty() or smile_cascade.empty():
        raise RuntimeError("Failed to load Haar cascades from cv2.data.haarcascades")
    return face_cascade, smile_cascade

# Phát hiện nụ cười trong khung hình BGR và trả về bản sao đã chú thích và cờ boolean.
def detect_smiles_bgr(frame_bgr: np.ndarray,face_scale: float = 1.1,face_neighbors: int = 5,smile_scale: float = 1.7,smile_neighbors: int = 22,min_face_size: int = 80,) -> Tuple[np.ndarray, bool]:
    face_cascade, smile_cascade = _load_classifiers()
    img = frame_bgr.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=face_scale, minNeighbors=face_neighbors, minSize=(min_face_size, min_face_size))
    smiling_any = False

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=smile_scale, minNeighbors=smile_neighbors)
        if len(smiles) > 0:
            smiling_any = True
            cv2.putText(img, "Smiling :D", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            # Vẽ hộp nụ cười đầu tiên để tránh lộn xộn
            (sx, sy, sw, sh) = smiles[0]
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)
        else:
            cv2.putText(img, "Not smiling", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return img, smiling_any
