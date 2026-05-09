import cv2 as cv
from capture.camera import abrir_camera
from tracking.hand_tracker import detectar_mao

camera = abrir_camera()

while True:

    status, frame = camera.read()

    if not status:
        break

    mao = detectar_mao(frame)

    # mostrar frame
    cv.imshow("Camera", mao)

    # verificar tecla q
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# liberar recursos
camera.release()
cv.destroyAllWindows()