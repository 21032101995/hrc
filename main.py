import cv2
from fingers_processing import HandTracker
import time
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

def main():

    cap = cv2.VideoCapture(0)
    rtde_c = RTDEControl("192.168.1.150")
    rtde_r = RTDEReceive("192.168.1.150")
    tracker = HandTracker("hand_landmarker.task")
    
    while True:

        ret, frame = cap.read()
        tcp_pose = rtde_r.getActualTCPPose()
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
            rtde_c.moveL([tcp_pose[0], tcp_pose[1], tcp_pose[2]+0.0001, tcp_pose[3],tcp_pose[4], tcp_pose[5]], 0.25, 0.5)
            

        
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