import cv2


class VideoReader:
    """A dedicated reader for video files or live camera streams using OpenCV."""

    def __init__(self, source: str | int):
        """
        Initialize the video reader.
        
        :param source: File path to a video (str) or camera index (int, e.g., 0).
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video source: {source}")


    def read_frame(self):
        """
        Reads the next frame.
        
        :return: (True, frame) if a frame was read, or (False, None) at EOF or error.
        """
        if not self.cap.isOpened():
            return False, None
        return self.cap.read()