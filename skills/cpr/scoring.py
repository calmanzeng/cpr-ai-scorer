"""
CPR 评分插件
实现 SkillScorer 接口，从手腕轨迹计算 CPR 质量
"""
import numpy as np
from collections import deque

# ---- 峰检测（纯 NumPy，零依赖）----
def find_peaks(data, min_dist=12, min_prominence=None):
    n = len(data); peaks = []
    for i in range(1, n - 1):
        if data[i] > data[i-1] and data[i] > data[i+1]:
            if peaks and i - peaks[-1] < min_dist:
                if data[i] > data[peaks[-1]]: peaks[-1] = i
                continue
            peaks.append(i)
    peaks = np.array(peaks, dtype=int)
    if min_prominence and len(peaks):
        fp = []
        for p in peaks:
            l, r = p, p
            while l > 0 and data[l] >= data[l-1]: l -= 1
            while r < n-1 and data[r] >= data[r+1]: r += 1
            if data[p] - max(data[l], data[r]) >= min_prominence: fp.append(p)
        peaks = np.array(fp, dtype=int)
    return peaks


class SkillScorer:
    """CPR 操作评分器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.metrics = config.get("metrics", [])
        self.feedback_rules = config.get("feedback", {}).get("rules", [])
        self.feature_history = {}
        self.fps = 30  # will be updated at runtime
    
    def extract_features(self, landmarks: dict, frame_idx: int, fps: float) -> dict:
        """从姿态关键点提取特征"""
        self.fps = fps
        
        # 提取关键坐标
        lw = landmarks.get("left_wrist", {})
        rw = landmarks.get("right_wrist", {})
        ls = landmarks.get("left_shoulder", {})
        rs = landmarks.get("right_shoulder", {})
        lh = landmarks.get("left_hip", {})
        rh = landmarks.get("right_hip", {})
        
        # 计算手腕 Y 轴归一化坐标
        wrist_y = (lw.get("y", 0) + rw.get("y", 0)) / 2
        shoulder_y = (ls.get("y", 0) + rs.get("y", 0)) / 2
        hip_y = (lh.get("y", 0) + rh.get("y", 0)) / 2
        torso = max(abs(shoulder_y - hip_y), 0.001)
        
        wyn = (wrist_y - shoulder_y) / torso
        
        return {"wrist_y_normalized": wyn}
    
    def compute_metrics(self, feature_history: dict) -> dict:
        """计算所有评分指标"""
        wyn_data = feature_history.get("wrist_y_normalized", [])
        
        if len(wyn_data) < 30:
            return {}
        
        y = np.array(wyn_data)
        
        # 去趋势
        if len(y) > 60:
            k = np.ones(min(60, len(y))) / min(60, len(y))
            yd = y - np.convolve(y, k, mode='same')
        else:
            yd = y - np.mean(y)
        
        ystd = np.std(yd)
        if ystd < 1e-6:
            return {}
        
        # 检测按压峰
        inv = -yd
        min_d = int(0.35 * self.fps)
        prom = ystd * 0.4
        peaks = find_peaks(inv, min_dist=min_d, min_prominence=prom)
        
        if len(peaks) < 3:
            return {}
        
        # ---- 计算指标 ----
        intervals = np.diff(peaks) / self.fps
        elapsed = (peaks[-1] - peaks[0]) / self.fps
        cpm = (len(peaks) - 1) / elapsed * 60 if elapsed > 0 else 0
        
        cv = (np.std(intervals) / np.mean(intervals) * 100
              if len(intervals) >= 2 and np.mean(intervals) > 0 else 0)
        
        depth = ystd * 2
        
        # ---- 评分 ----
        rate_score = max(0.0, min(1.0, 1.0 - abs(cpm - 110) / 35))
        
        if cv <= 10: cons_score = 1.0
        elif cv <= 15: cons_score = 0.8
        elif cv <= 25: cons_score = 0.5
        else: cons_score = 0.2
        
        depth_score = min(1.0, depth / 0.3) if depth > 0 else 0
        
        overall = int((rate_score * 0.5 + cons_score * 0.3 + depth_score * 0.2) * 100)
        
        return {
            "compression_rate": {"value": cpm, "unit": "CPM", "score": rate_score},
            "rhythm_consistency": {"value": cv, "unit": "%", "score": cons_score},
            "depth_index": {"value": depth, "unit": "", "score": depth_score},
            "overall": {"value": overall, "unit": "/100", "score": overall / 100},
            "compression_count": {"value": len(peaks), "unit": "次", "score": 1.0},
        }
    
    def get_feedback(self, metrics: dict) -> list:
        """根据指标生成反馈"""
        feedback = []
        
        rate = metrics.get("compression_rate", {}).get("value", 0)
        cv = metrics.get("rhythm_consistency", {}).get("value", 0)
        
        if rate < 100: feedback.append({"severity": "warning", "message": f"按压太慢 ({rate:.0f}/min) 加快！"})
        elif rate > 120: feedback.append({"severity": "warning", "message": f"按压太快 ({rate:.0f}/min) 放慢！"})
        else: feedback.append({"severity": "success", "message": f"频率正常 ({rate:.0f}/min)"})
        
        if cv > 25: feedback.append({"severity": "error", "message": f"节奏严重不均匀 (CV={cv:.0f}%)"})
        elif cv > 15: feedback.append({"severity": "warning", "message": f"节奏可改进 (CV={cv:.0f}%)"})
        else: feedback.append({"severity": "success", "message": f"节奏稳定 (CV={cv:.0f}%)"})
        
        return feedback
