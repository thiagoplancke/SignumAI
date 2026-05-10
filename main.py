import cv2 as cv

from capture.camera import abrir_camera
from tracking.hand_tracker import detectar_mao
from dataset.collect import collect_sample


camera = abrir_camera()

print("=== SIGNUM AI ===")
print("Pressione A-Z para iniciar coleta")
print("Pressione 1 para sair")


# ----------------------------
# ESTADO DA COLETA
# ----------------------------
collecting = False
current_label = None
samples_left = 0

frame_counter = 0
save_interval = 5  # salva a cada 5 frames


while True:

    status, frame = camera.read()

    if not status:
        break

    # ----------------------------
    # DETECÇÃO
    # ----------------------------
    frame, hand_landmarks = detectar_mao(frame)

    # ----------------------------
    # TECLADO
    # ----------------------------
    key = cv.waitKey(1) & 0xFF

    # ----------------------------
    # INICIAR COLETA
    # ----------------------------
    if 97 <= key <= 122:

        current_label = chr(key).upper()

        collecting = True
        samples_left = 50

        print(f"\n[COLETANDO] letra {current_label}")

    # ----------------------------
    # COLETA AUTOMÁTICA
    # ----------------------------
    if collecting and len(hand_landmarks) > 0:

        frame_counter += 1

        # pega primeira mão
        mao = hand_landmarks[0]

        # salva só a cada N frames
        if frame_counter % save_interval == 0:

            collect_sample(mao, current_label)

            samples_left -= 1

            print(f"faltam: {samples_left}")

        # terminou coleta
        if samples_left <= 0:

            collecting = False

            print(f"[FINALIZADO] letra {current_label}\n")

    # ----------------------------
    # MOSTRAR CÂMERA
    # ----------------------------
    cv.imshow("SignumAI", frame)

    # ----------------------------
    # SAIR
    # ----------------------------
    if key == ord('1'):
        break


camera.release()
cv.destroyAllWindows()