import numpy as np
from sort import Sort

class FlowTracker:
    def __init__(self, frame_width):
        from sort import Sort
        self.tracker = Sort()
        self.line_x = frame_width // 2   # 中心竖线
        self.enter_count = 0             # 左→右
        self.exit_count  = 0             # 右→左
        self.pre_positions = {}          # track_id -> last center x

    def update(self, boxes_xyxy):
        # 同前，把 (N,4) 加上 dummy score，返回 tracks
        dets = np.hstack([boxes_xyxy, np.zeros((len(boxes_xyxy),1))])
        return self.tracker.update(dets)

    def count(self, tracks):
        """
        tracks: np.ndarray of shape (M,5) 每行 [x1,y1,x2,y2,track_id]
        """
        for x1,y1,x2,y2,tid in tracks:
            cx = int((x1 + x2) / 2)  # 中心点 x 坐标
            if tid in self.pre_positions:
                prev = self.pre_positions[tid]
                # 从左到右 → 计入 enter_count；从右到左 → 计入 exit_count
                if prev < self.line_x <= cx:
                    self.enter_count += 1
                elif prev > self.line_x >= cx:
                    self.exit_count += 1
            self.pre_positions[tid] = cx
        return self.enter_count, self.exit_count
