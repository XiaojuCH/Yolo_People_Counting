import cv2
import torch
import numpy as np
from ultralytics import YOLO
from camera import Camera
# 如果不需要进出计数，可注释掉下面两行
from tracker import FlowTracker

class FlowDetector:
    def __init__(self,
                 model_path="models/yolov8n.pt",
                 use_gpu=False,
                 cam_idx=0,
                 width=640,
                 height=480):
        # 选择设备
        self.device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        # 加载模型
        self.model = YOLO(model_path).to(self.device)
        # 摄像头
        self.cam = Camera(idx=cam_idx, width=width, height=height)
        # 进出计数器
        self.tracker = FlowTracker(frame_width=width)
    def run(self):
        while True:
            frame = self.cam.read()
            results = self.model(frame, device=self.device, verbose=False)[0]
            boxes = results.boxes.xyxy.cpu().numpy()[results.boxes.cls.cpu().numpy()==0]

            tracks = self.tracker.update(boxes)
            enter, exit_ = self.tracker.count(tracks)

            annotated = results.plot()
            h, w = frame.shape[:2]
            # 画竖线
            cv2.line(annotated, (w//2, 0), (w//2, h), (0,255,255), 2)
            # 左→右 统计
            cv2.putText(annotated, f"L->R: {enter}", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            # 右→左 统计
            cv2.putText(annotated, f"R->L: {exit_}", (10,90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            cv2.imshow("FlowDetector", annotated)
            if cv2.waitKey(1) == 27:
                break

        self.cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 若无 NVIDIA 显卡，use_gpu=False；有显卡且要跑 GPU 推理才设成 True
    FlowDetector(use_gpu=False).run()
