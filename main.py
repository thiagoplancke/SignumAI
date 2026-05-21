import cv2 as cv

from capture.camera import abrir_camera
from dataset.collect import collect_sample
from model.predict import testar_modelo
from tracking.hand_tracker import detectar_mao


WINDOW_NAME = "SignumAI"

MODE_PREDICT = "predict"
MODE_COLLECT = "collect"

SAMPLES_PER_LABEL = 50
SAVE_INTERVAL = 5


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


def draw_header(frame, mode):
    draw_text(frame, f"Modo: {mode.upper()}", (20, 40))
    draw_text(
        frame,
        "C = Coleta | P = Predicao | 1 = Sair",
        (20, 80),
        scale=0.6,
        color=(200, 200, 200)
    )


def handle_controls(key, state):
    if key == ord("p"):
        state["mode"] = MODE_PREDICT
        state["collecting"] = False
        print("\n[MODO PREDICAO]")

    elif key == ord("c"):
        state["mode"] = MODE_COLLECT
        state["collecting"] = False
        print("\n[MODO COLETA]")

    elif key == ord("1"):
        state["running"] = False


def handle_collect_mode(frame, hand_landmarks, key, state):
    if 97 <= key <= 122:
        state["current_label"] = chr(key).upper()
        state["collecting"] = True
        state["samples_left"] = SAMPLES_PER_LABEL
        state["frame_counter"] = 0
        print(f"\n[COLETANDO] {state['current_label']}")

    if not state["collecting"]:
        draw_text(
            frame,
            "Pressione A-Z para coletar",
            (20, 130),
            color=(0, 255, 255)
        )
        return

    state["frame_counter"] += 1

    draw_text(
        frame,
        f"Coletando: {state['current_label']}",
        (20, 130),
        color=(0, 255, 255)
    )
    draw_text(
        frame,
        f"Faltam: {state['samples_left']}",
        (20, 170),
        color=(0, 255, 255)
    )

    if state["frame_counter"] % SAVE_INTERVAL == 0:
        collect_sample(hand_landmarks, state["current_label"])
        state["samples_left"] -= 1

    if state["samples_left"] <= 0:
        state["collecting"] = False
        print(f"[FINALIZADO] {state['current_label']}")


def handle_predict_mode(frame, hand_landmarks):
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


def main():
    camera = abrir_camera()

    state = {
        "mode": MODE_PREDICT,
        "running": True,
        "collecting": False,
        "current_label": None,
        "samples_left": 0,
        "frame_counter": 0,
    }

    print("=== SIGNUM AI ===")
    print("[MODO PREDICAO]")

    while state["running"]:
        status, frame = camera.read()

        if not status:
            break

        frame, all_hand_landmarks = detectar_mao(frame)

        key = cv.waitKey(1) & 0xFF
        handle_controls(key, state)

        draw_header(frame, state["mode"])

        if all_hand_landmarks:
            hand_landmarks = all_hand_landmarks[0]

            if state["mode"] == MODE_COLLECT:
                handle_collect_mode(frame, hand_landmarks, key, state)

            elif state["mode"] == MODE_PREDICT:
                handle_predict_mode(frame, hand_landmarks)
        else:
            draw_text(frame, "Nenhuma mao detectada", (20, 130), color=(0, 0, 255))

        cv.imshow(WINDOW_NAME, frame)

    camera.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
