"""一键演示脚本 — 适合面试/BD场景直接运行"""
import subprocess, sys, os, time

def main():
    # Check deps
    try:
        import cv2, numpy, mediapipe
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run: pip install mediapipe opencv-python numpy")
        return 1

    model = os.path.expanduser("~/.hermes/cache/pose_landmarker_lite.task")
    if not os.path.exists(model):
        print(f"Model not found: {model}")
        print("Download from Google Storage and save to that path.")
        return 1

    print("""
    ╔══════════════════════════════════════════════╗
    ║   执医24项 AI评分系统 — CPR 演示模式         ║
    ║                                              ║
    ║   按 Q: 退出                                 ║
    ║   按 R: 重置评分                             ║
    ║   按 S: 保存本次评估报告                      ║
    ║                                              ║
    ║   演示要点:                                  ║
    ║   1. 对着摄像头做 CPR (双手叠放在胸前按压)    ║
    ║   2. 观察实时评分变化                         ║
    ║   3. 故意做快/慢/乱 → 看评分实时反馈       ║
    ╚══════════════════════════════════════════════╝
    """)

    mvp = os.path.join(os.path.dirname(__file__), "mvp.py")
    return subprocess.run([sys.executable, mvp, "--report", "cpr_demo_report.json"]).returncode

if __name__ == "__main__":
    exit(main())
