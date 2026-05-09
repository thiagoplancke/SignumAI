import cv2 as cv

def abrir_camera():
    return cv.VideoCapture(0)