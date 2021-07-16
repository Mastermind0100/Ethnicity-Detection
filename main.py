from deepface import DeepFace
import cv2
import math

def predictor(face_detected):
    try:
        pred = DeepFace.analyze(img_path=face_detected, actions=['race'])
        return pred['dominant_race']
    except ValueError:
        return None


def find_aoi(faces_list, x_lim, y_lim):
    new_faces = []
    for (x_i, y_i, w_i, h_i) in faces_list:
        inc_margin = int(0.4*(math.dist((x_i, y_i), (x_i+w_i, y_i+h_i))))
        y_n = y_i - inc_margin if y_i > inc_margin else 0
        h_n = h_i + inc_margin if y_i+h_i+inc_margin <= y_lim else y_lim
        x_n = x_i - inc_margin if x_i > inc_margin else 0
        w_n = w_i + inc_margin if x_i+w_i+inc_margin <= x_lim else x_lim
        new_faces.append((x_n, y_n, w_n, h_n))
    return new_faces


if __name__ == "__main__":
    img = cv2.imread("test_imgs/men.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('res/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    alt_faces = find_aoi(faces, img.shape[1], img.shape[0])
    for (x, y, w, h) in alt_faces:
        predictor(img[y:y + h, x:x + w])
        # cv2.imshow('window', img[y:y+h, x:x+w])
        # cv2.waitKey(0)
