import cv2 as cv

from capture.camera import abrir_camera
from model.predict import testar_modelo
from tracking.hand_tracker import detectar_mao


WINDOW_NAME = "SignumAI"


def draw_text(frame, text, position, scale=0.8, color=(255, 255, 255)):
    cv.putText(
        frame,
        text,
        position,
        cv.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2
    )


def draw_header(frame):
    draw_text(frame, "Modo: PREDICAO", (20, 40))
    draw_text(
        frame,
        "1 = Sair",
        (20, 80),
        scale=0.6,
        color=(200, 200, 200)
    )


def predict_frame(frame, hand_landmarks):
    letra, confidence = testar_modelo(hand_landmarks)

    draw_text(
        frame,
        f"Letra: {letra}",
        (20, 130),
        scale=1,
        color=(0, 255, 0)
    )
    draw_text(
        frame,
        f"Conf: {confidence:.2f}",
        (20, 170),
        color=(255, 255, 0)
    )


def wait_start_command():
    print("=== SIGNUM AI ===")
    print("Pressione P para iniciar a predicao.")

    command = input("> ").strip().lower()
    return command == "p"


def main():
    if not wait_start_command():
        print("Predicao cancelada.")
        return

    camera = abrir_camera()
    print("[MODO PREDICAO]")

    while True:
        status, frame = camera.read()

        if not status:
            break

        frame, all_hand_landmarks = detectar_mao(frame)
        key = cv.waitKey(1) & 0xFF

        if key == ord("1"):
            break

        draw_header(frame)

        if all_hand_landmarks:
            predict_frame(frame, all_hand_landmarks[0])
        else:
            draw_text(frame, "Nenhuma mao detectada", (20, 130), color=(0, 0, 255))

        cv.imshow(WINDOW_NAME, frame)

    camera.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
