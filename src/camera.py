import cv2

class Camera:
    def __init__(self, idx=0, width=640, height=480, buffersize=3):
        self.cap = cv2.VideoCapture(idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffersize)
        if not self.cap.isOpened():
            raise RuntimeError(f"打开摄像头 {idx} 失败")

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("读取摄像头帧失败")
        return frame

    def release(self):
        self.cap.release()
