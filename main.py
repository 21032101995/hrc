import cv2
from fingers_processing import HandTracker


def main():

    cap = cv2.VideoCapture(0)

    tracker = HandTracker("hand_landmarker.task")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Espelhar a imagem da webcam
        frame = cv2.flip(frame, 1)

        # Processar frame
        tracker.process_frame(frame)

        # Verificar se o indicador da mão direita
        # está verticalmente apontado para baixo
        if tracker.is_right_index_pointing_down():
            print("INDICADOR DIREITO APONTADO PARA BAIXO")

            cv2.putText(
                frame,
                "INDICADOR PARA BAIXO",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        else:
            cv2.putText(
                frame,
                "Indicador nao esta para baixo",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        if tracker.is_right_index_pointing_up():
            print("INDICADOR DIREITO APONTADO PARA CIMA")
        
            cv2.putText(
                frame,
                "INDICADOR PARA CIMA",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
            )
        
        else:
            cv2.putText(
                frame,
                "Indicador nao esta para cima",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )


        # Mostrar imagem
        cv2.imshow("Hand Tracker", frame)

        # Pressionar Q para sair
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()