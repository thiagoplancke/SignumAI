import cv2 as cv
from capture.camera import abrir_camera
from tracking.hand_tracker import detectar_mao
from geometry.angulo import calcular_angulo
from utils.smoothing import media_angulos


camera = abrir_camera()

while True:

    status, frame = camera.read()

    if not status:
        break

    frame,hand_landmarks = detectar_mao(frame)
    if len(hand_landmarks) > 0:
        p5 = hand_landmarks[0][5]
        p6 = hand_landmarks[0][6]
        p8 = hand_landmarks[0][8]
        angulo = calcular_angulo([p5.x, p5.y], [p6.x, p6.y], [p8.x, p8.y])
        if angulo < 10:
            print("dedo fechado")
        if angulo > 160:
            print("dedo aberto")
        if angulo >= 10 and angulo <= 160:
            print("dedo meio aberto")            

    # mostrar frame
    cv.imshow("Camera", frame)

    # verificar tecla q
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# liberar recursos
camera.release()
cv.destroyAllWindows()