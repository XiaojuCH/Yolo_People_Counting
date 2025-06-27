import numpy as np
from sort import Sort

class FlowTracker:
    def __init__(self, frame_height):
        self.tracker = Sort()
        self.line_y = frame_height // 2
        self.enter_count = 0
        self.exit_count = 0
        self.pre_positions = {}  # track_id -> last center y

    def update(self, boxes_xyxy):
        """
        boxes_xyxy: np.ndarray of shape (N,4) 每行 [x1,y1,x2,y2]
        """
        dets = np.hstack([boxes_xyxy, np.zeros((len(boxes_xyxy),1))])  # dummy score
        tracks = self.tracker.update(dets)
        return tracks

    def count(self, tracks):
        """
        tracks: np.ndarray of shape (M,5) 每行 [x1,y1,x2,y2,track_id]
        """
        for x1,y1,x2,y2,tid in tracks:
            cy = int((y1+y2)/2)
            if tid in self.pre_positions:
                prev = self.pre_positions[tid]
                # 从上到下→进入；从下到上→离开
                if prev < self.line_y <= cy:
                    self.exit_count += 1
                elif prev > self.line_y >= cy:
                    self.enter_count += 1
            self.pre_positions[tid] = cy
        return self.enter_count, self.exit_count
