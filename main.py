import cv2 as cv

from capture.camera import abrir_camera
from tracking.hand_tracker import detectar_mao

from dataset.collect import collect_sample
from model.predict import testar_modelo


camera = abrir_camera()

print("=== SIGNUM AI ===")


# ----------------------------
# ESTADOS
# ----------------------------
modo = "predict"

collecting = False
current_label = None
samples_left = 0

frame_counter = 0
save_interval = 5


# ----------------------------
# LOOP PRINCIPAL
# ----------------------------
while True:

    status, frame = camera.read()

    if not status:
        break

    frame, hand_landmarks = detectar_mao(frame)

    key = cv.waitKey(1) & 0xFF

    # ==================================================
    # CONTROLES
    # ==================================================
    

    if key == ord('p'):

        modo = "predict"

        print("\n[MODO PREDIÇÃO]")

    elif key == ord('1'):

        break

    # ==================================================
    # TEXOS NA TELA
    # ==================================================
    cv.putText(
        frame,
        f"Modo: {modo.upper()}",
        (20, 40),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv.putText(
        frame,
        " P = Predicao | 1 = Sair",
        (20, 80),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        2
    )

    # ==================================================
    # EXISTE MÃO?
    # ==================================================
    if len(hand_landmarks) > 0:

        mao = hand_landmarks[0]

        # ==================================================
        # MODO COLETA
        # ==================================================
        if modo == "collect":

            # inicia coleta
            if 97 <= key <= 122:

                current_label = chr(key).upper()

                collecting = True

                samples_left = 50

                print(f"\n[COLETANDO] {current_label}")

            # coleta automática
            if collecting:

                frame_counter += 1

                cv.putText(
                    frame,
                    f"Coletando: {current_label}",
                    (20, 130),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                cv.putText(
                    frame,
                    f"Faltam: {samples_left}",
                    (20, 170),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                if frame_counter % save_interval == 0:

                    collect_sample(mao, current_label)

                    samples_left -= 1

                if samples_left <= 0:

                    collecting = False

                    print(f"[FINALIZADO] {current_label}")

        # ==================================================
        # MODO PREDIÇÃO
        # ==================================================
        elif modo == "predict":

            letra, confidence = testar_modelo(mao)

            cv.putText(
                frame,
                f"Letra: {letra}",
                (20, 130),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv.putText(
                frame,
                f"Conf: {confidence:.2f}",
                (20, 170),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

    # ==================================================
    # MOSTRAR FRAME
    # ==================================================
    cv.imshow("SignumAI", frame)


camera.release()
cv.destroyAllWindows()