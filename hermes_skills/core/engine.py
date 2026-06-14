"""
技能训练 Pipeline 引擎
编排: 输入 → AI推理 → 特征提取 → 评分 → LLM导师 → 渲染 → 数据记录
"""
import time
from collections import deque
import numpy as np
import cv2

from .registry import SkillRegistry
from .data_hub import DataHub
from ..ai.llm_tutor import LLMTutor, OfflineTutor


class SkillPipeline:
    """执医技能训练流水线"""
    
    def __init__(self, skill_name: str, skills_dir: str = "./skills",
                 data_hub: DataHub = None,
                 llm_provider: str = None, llm_api_key: str = None,
                 llm_model: str = None, llm_base_url: str = None):
        self.skill_name = skill_name
        self.skills_dir = skills_dir
        self.data_hub = data_hub or DataHub()
        
        # LLM 导师
        if llm_provider and llm_provider != "none":
            self.llm_tutor = LLMTutor(
                provider=llm_provider, api_key=llm_api_key,
                model=llm_model, base_url=llm_base_url
            )
        else:
            self.llm_tutor = None
        
        self._metrics_history = []
        self._llm_feedback = ""
        
        # 加载技能
        registry = SkillRegistry(skills_dir)
        skill = registry.get_skill(skill_name)
        self.config = skill["config"]
        self.scorer = skill["scorer"]
        
        if self.scorer is None:
            raise RuntimeError(f"技能 '{skill_name}' 缺少 scoring.py")
        
        self._running = False
        self._session_id = None
        self._frame_count = 0
        self._latest_metrics = None
        self._fps = 30
        self._fps_counter = deque(maxlen=30)
        self._last_tick = time.time()
    
    def _open_source(self, camera=0, video_path=None):
        if video_path:
            cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(camera)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源")
        self._fps = cap.get(cv2.CAP_PROP_FPS)
        if self._fps <= 0:
            self._fps = 30
        return cap
    
    def _tick(self):
        now = time.time()
        dt = now - self._last_tick
        self._last_tick = now
        if dt > 0:
            self._fps_counter.append(1.0 / dt)
    
    def run(self, camera=0, video_path=None, headless=False,
            on_frame=None, on_metrics=None):
        cap = self._open_source(camera, video_path)
        self._session_id = self.data_hub.start_session(self.skill_name)
        self._running = True
        self._frame_count = 0
        self._metrics_history = []
        self._llm_feedback = ""
        
        ai_backend = self._init_ai_backend()
        
        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self._frame_count += 1
                self._tick()
                
                # 1. AI 推理
                landmarks = ai_backend.detect(frame)
                
                # 2. 特征提取
                if landmarks:
                    features = self.scorer.extract_features(
                        landmarks, self._frame_count, self._fps
                    )
                    self.data_hub.add_features(features)
                
                # 3. 评分
                feature_history = self.data_hub.get_feature_history()
                metrics = self.scorer.compute_metrics(feature_history)
                
                if metrics:
                    self._latest_metrics = metrics
                    self._metrics_history.append(metrics)
                    if len(self._metrics_history) > 30:
                        self._metrics_history.pop(0)
                    self.data_hub.save_metrics(self._session_id, metrics)
                    if on_metrics:
                        on_metrics(metrics)
                    
                    # LLM 导师反馈
                    if self.llm_tutor and self.llm_tutor.is_enabled():
                        llm_fb = self.llm_tutor.get_feedback(
                            self.skill_name, metrics, self._metrics_history
                        )
                        if llm_fb:
                            self._llm_feedback = llm_fb
                    elif self._frame_count % 90 == 0:
                        self._llm_feedback = OfflineTutor.get_feedback(
                            self.skill_name, metrics, self._metrics_history
                        )
                
                # 4. 渲染
                fb = self.scorer.get_feedback(metrics) if metrics else []
                tutor_fb = self._llm_feedback
                
                if on_frame:
                    frame = on_frame(frame, metrics, landmarks, fb, tutor_fb,
                                     list(feature_history.get("wrist_y_normalized", [])))
                
                # 5. 显示
                if not headless:
                    cv2.imshow(f"执医 AI - {self.config['display_name']}", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        self.reset()
        
        finally:
            cap.release()
            if not headless:
                cv2.destroyAllWindows()
            self.data_hub.end_session(self._session_id, {
                "avg_score": self._latest_metrics.get("overall", {}).get("value")
                if self._latest_metrics else None,
                "frames": self._frame_count,
            })
        
        return self._session_id
    
    def _init_ai_backend(self):
        from hermes_skills.ai.pose_landmarker import PoseLandmarkerBackend
        model_type = self.config.get("model", {}).get("type", "mediapipe_pose")
        model_path = self.config.get("model", {}).get("model_path", "")
        confidence = self.config.get("model", {}).get("min_detection_confidence", 0.5)
        if model_type == "mediapipe_pose":
            return PoseLandmarkerBackend(model_path, confidence)
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    def reset(self):
        self._session_id = self.data_hub.start_session(self.skill_name)
        self._frame_count = 0
        self._latest_metrics = None
        self._metrics_history = []
        self._llm_feedback = ""
        print("🔄 评分已重置")
    
    def stop(self):
        self._running = False
    
    @property
    def metrics(self):
        return self._latest_metrics
    
    @property
    def fps(self):
        return np.mean(self._fps_counter) if self._fps_counter else 0
