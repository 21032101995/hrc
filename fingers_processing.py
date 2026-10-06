import cv2
import mediapipe as mp

class HandTracker:

    def __init__(self, model_path="hand_landmarker.task"):
        base_options = mp.tasks.BaseOptions(
            model_asset_path=model_path
        )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.detector = (
            mp.tasks.vision.HandLandmarker
            .create_from_options(options)
        )

        self.results = None

    def process_frame(self, frame):
        """
        Recebe um frame OpenCV (BGR) e processa-o.
        """

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        self.results = self.detector.detect(image)

        return self.results

    def get_hands_landmarks(self):
        """
        Retorna os landmarks de todas as mãos.
        """

        if self.results is None:
            return []

        return self.results.hand_landmarks

    def get_hand(self, hand_index=0):
        """
        Retorna os landmarks de uma mão específica.
        """

        hands = self.get_hands_landmarks()

        if hand_index >= len(hands):
            return None

        return hands[hand_index]

    def get_fingers(self, hand_index=0):

        hand = self.get_hand(hand_index)

        if hand is None:
            return {}

        fingers = {}

        # -------------------------
        # Polegar
        # -------------------------

        thumb_tip = hand[4]
        thumb_ip = hand[3]

        fingers["thumb"] = thumb_tip.x < thumb_ip.x

        # -------------------------
        # Indicador
        # -------------------------

        fingers["index"] = (
            hand[8].y < hand[6].y
        )

        # -------------------------
        # Médio
        # -------------------------

        fingers["middle"] = (
            hand[12].y < hand[10].y
        )

        # -------------------------
        # Anelar
        # -------------------------

        fingers["ring"] = (
            hand[16].y < hand[14].y
        )

        # -------------------------
        # Mindinho
        # -------------------------

        fingers["pinky"] = (
            hand[20].y < hand[18].y
        )

        return fingers

    def count_fingers(self, hand_index=0):
        """
        Retorna o número de dedos levantados.
        """

        fingers = self.get_fingers(hand_index)

        return sum(fingers.values())

    def is_right_index_pointing_down(self, hand_index=0):

        hand = self.get_hand(hand_index)

        if hand is None:
            return False

        # Pontos do indicador
        pip = hand[6]
        dip = hand[7]
        tip = hand[8]

        # O dedo está a apontar para baixo
        pointing_down = (
        tip.y > dip.y > pip.y
        )

        return pointing_down

    def is_right_index_pointing_up(self, hand_index=0):
        hand = self.get_hand(hand_index)
        
        if hand is None:
            return False
        
        # Pontos do indicador
        pip = hand[6]
        dip = hand[7]
        tip = hand[8]
        
        # O dedo está a apontar para baixo
        pointing_up = (
            tip.y < dip.y < pip.y
        )
        
        return pointing_up

    def close(self):
        self.detector.close()