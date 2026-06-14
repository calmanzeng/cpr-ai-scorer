"""
AI 导师模块 — 大模型驱动的个性化教学反馈
=========================================
支持: OpenAI / DeepSeek / Ollama 本地模型 / Hermes 内置模型
职责: 接收数值评分 → 生成自然语言教学指导
"""

import json
import time
from typing import Optional
from collections import deque


# ──── Prompt 模板 ────

CPR_TUTOR_PROMPT = """你是一位执医技能考试的资深考官，正在指导一位住院医师做心肺复苏（CPR）操作。

## 当前实时数据
- 按压频率: {rate} CPM（标准: 100-120 CPM）
- 节奏一致性: {cv}%（越低越好，<15% 为优秀）
- 深度指数: {depth:.3f}（越高越有力）
- 整体评分: {overall}/100
- 当前检测到的按压次数: {count}

## 近期趋势（过去 30 秒）
{trend_context}

## 要求
请用中文给出简洁的教学反馈（2-3句话），格式如下：
1. 先评价当前表现（如果评分>=80，先表扬）
2. 指出最需要改进的 1 个具体问题
3. 给出可操作的改进建议

注意：
- 语气像考官一样专业但亲切
- 不要重复数据本身，要说数据意味着什么
- 每次反馈控制在 60 字以内
- 如果数据表明操作良好，就说继续保持

## 输出格式
只输出反馈文本，不要前缀、编号或额外解释。"""


GENERIC_TUTOR_PROMPT = """你是一位执医技能考试的资深考官，正在用 AI 系统评估一位住院医师的操作。

## 技能: {skill_name}
## 当前评分数据
{metrics_table}

## 要求
请用中文给出简洁的教学反馈（2-3句话）：
1. 评价当前整体表现
2. 指出最需改进的具体问题
3. 给出可操作的改进建议

语气专业但鼓励，每次 60 字以内。只输出反馈文本。"""


# ──── LLM 后端适配器 ────

class LLMBackend:
    """统一的 LLM 调用接口，适配不同后端"""
    
    def __init__(self, provider: str = "openai", api_key: str = None,
                 model: str = None, base_url: str = None):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        
        # 默认模型
        if not self.model:
            defaults = {
                "openai": "gpt-4o-mini",
                "deepseek": "deepseek-chat",
                "ollama": "qwen2.5:7b",
                "openai_compatible": "default",
            }
            self.model = defaults.get(provider, "gpt-4o-mini")
    
    def chat(self, prompt: str, max_tokens: int = 200) -> Optional[str]:
        """调用 LLM 获取回复"""
        
        if self.provider in ("openai", "deepseek", "openai_compatible"):
            return self._call_openai_api(prompt, max_tokens)
        elif self.provider == "ollama":
            return self._call_ollama(prompt, max_tokens)
        else:
            raise ValueError(f"不支持的 LLM provider: {self.provider}")
    
    def _call_openai_api(self, prompt: str, max_tokens: int) -> Optional[str]:
        """调用 OpenAI 兼容 API"""
        import urllib.request
        
        if self.provider == "deepseek":
            api_url = self.base_url or "https://api.deepseek.com/v1/chat/completions"
        elif self.provider == "openai_compatible":
            api_url = self.base_url or "http://localhost:8080/v1/chat/completions"
        else:
            api_url = self.base_url or "https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是一位专业、鼓励型的执医技能考官。回复简洁，60字以内。"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        try:
            req = urllib.request.Request(
                api_url,
                data=json.dumps(payload).encode(),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read())
                return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"  [LLM Error] {e}")
            return None
    
    def _call_ollama(self, prompt: str, max_tokens: int) -> Optional[str]:
        """调用本地 Ollama"""
        import urllib.request
        
        url = self.base_url or "http://localhost:11434/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.7}
        }
        
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read())
                return result.get("response", "").strip()
        except Exception as e:
            return None


# ──── AI 导师 ────

class LLMTutor:
    """
    AI 教学导师
    
    用法:
        tutor = LLMTutor()
        feedback = tutor.get_feedback(skill="cpr", metrics={...}, history=[...])
    """
    
    def __init__(self, provider: str = None, api_key: str = None,
                 model: str = None, base_url: str = None,
                 min_interval: float = 8.0):
        """
        Args:
            provider: openai | deepseek | ollama | openai_compatible | None(disabled)
            api_key: API 密钥
            model: 模型名
            base_url: 自定义 API 地址
            min_interval: 最小调用间隔（秒），避免频繁调用产生费用
        """
        self.enabled = provider is not None and provider != "none"
        self.min_interval = min_interval
        self._last_call = 0
        self._last_feedback = ""
        self._feedback_cache = {}
        
        if self.enabled:
            self.backend = LLMBackend(
                provider=provider, api_key=api_key,
                model=model, base_url=base_url
            )
        else:
            self.backend = None
    
    def _should_call(self) -> bool:
        """检查是否应该调用 LLM（节流）"""
        if not self.enabled:
            return False
        return time.time() - self._last_call >= self.min_interval
    
    def _build_trend_context(self, history: list) -> str:
        """从指标历史中构建趋势描述"""
        if not history or len(history) < 3:
            return "暂无足够趋势数据"
        
        lines = []
        # 分析最近 5 个数据点
        recent = history[-5:]
        
        scores = [h.get("overall", {}).get("value", 0) for h in recent]
        rates = [h.get("compression_rate", {}).get("value", 0) for h in recent]
        cvs = [h.get("rhythm_consistency", {}).get("value", 0) for h in recent]
        
        if scores:
            trend = "上升" if scores[-1] > scores[0] else ("下降" if scores[-1] < scores[0] else "稳定")
            lines.append(f"- 评分趋势: {trend}（从 {scores[0]:.0f} 到 {scores[-1]:.0f}）")
        
        if rates:
            lines.append(f"- 频率范围: {min(rates):.0f} ~ {max(rates):.0f} CPM")
        
        return "\n".join(lines) if lines else "暂无趋势数据"
    
    def get_feedback(self, skill: str, metrics: dict,
                     history: list = None) -> Optional[str]:
        """
        获取 AI 导师反馈
        
        Args:
            skill: 技能名称
            metrics: 当前指标
            history: 历史指标列表
        
        Returns:
            AI 生成的反馈文本，或 None（未启用/节流中/调用失败）
        """
        if not self._should_call() or not metrics:
            return self._last_feedback if self._last_feedback else None
        
        # 构建 prompt
        if skill == "cpr":
            rate = metrics.get("compression_rate", {}).get("value", 0)
            cv = metrics.get("rhythm_consistency", {}).get("value", 0)
            depth = metrics.get("depth_index", {}).get("value", 0)
            overall = metrics.get("overall", {}).get("value", 0)
            count = metrics.get("compression_count", {}).get("value", 0)
            
            trend = self._build_trend_context(history or [])
            
            prompt = CPR_TUTOR_PROMPT.format(
                rate=f"{rate:.0f}",
                cv=f"{cv:.1f}",
                depth=depth,
                overall=int(overall),
                count=int(count),
                trend_context=trend
            )
        else:
            rows = []
            for key, val in metrics.items():
                rows.append(f"- {key}: {val.get('value', val)} {val.get('unit', '')}")
            prompt = GENERIC_TUTOR_PROMPT.format(
                skill_name=skill,
                metrics_table="\n".join(rows)
            )
        
        # 调用 LLM
        self._last_call = time.time()
        result = self.backend.chat(prompt)
        
        if result:
            self._last_feedback = result
        
        return result
    
    def is_enabled(self) -> bool:
        return self.enabled


# ──── 离线导师（无需 API 的备选方案）────

class OfflineTutor:
    """
    离线规则导师 — 不需要 LLM API，基于规则生成个性化反馈。
    当 LLM 不可用时自动降级使用。
    """
    
    @staticmethod
    def get_feedback(skill: str, metrics: dict, history: list = None) -> str:
        if skill == "cpr":
            return OfflineTutor._cpr_feedback(metrics, history)
        return OfflineTutor._generic_feedback(metrics)
    
    @staticmethod
    def _cpr_feedback(metrics: dict, history: list = None) -> str:
        overall = metrics.get("overall", {}).get("value", 0)
        rate = metrics.get("compression_rate", {}).get("value", 0)
        cv = metrics.get("rhythm_consistency", {}).get("value", 0)
        
        parts = []
        
        # Opening
        if overall >= 85:
            parts.append("表现优秀，继续保持！")
        elif overall >= 70:
            parts.append("整体不错，还有提升空间。")
        else:
            parts.append("需要加强练习，重点关注以下问题。")
        
        # Rate
        if rate < 95:
            parts.append(f"频率偏低（{rate:.0f}），试着跟随节拍器节奏。")
        elif rate > 125:
            parts.append(f"频率偏高（{rate:.0f}），放慢速度，确保每次按压充分回弹。")
        
        # Consistency
        if cv > 20:
            parts.append(f"按压节奏不够均匀，试着保持稳定的力量输出。")
        
        return " ".join(parts[:2])  # 最多 2 句
    
    @staticmethod
    def _generic_feedback(metrics: dict) -> str:
        scores = [m.get("score", m.get("value", 0)) for m in metrics.values()
                  if isinstance(m, dict)]
        avg = sum(scores) / len(scores) if scores else 0
        
        if avg >= 0.8:
            return "操作规范，继续保持。"
        elif avg >= 0.6:
            return "整体尚可，注意细节改进。"
        else:
            return "请参照标准流程加强练习。"
