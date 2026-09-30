"""
Distillation으로 학습한 student 모델을 FP16 / INT8로 각각 export하고,
정확도(mAP50)와 속도(추론 시간)를 비교한다.

주의: 여기서 만든 .engine 파일은 이 PC(RTX 3050)에서만 유효합니다.
      실제 Jetson 배포용 엔진은 Jetson 장비에서 다시 export해야 합니다.
      이 스크립트는 "FP32 vs FP16 vs INT8 중 뭐가 유리한가"를 검증하는 용도입니다.

사용법:
    python export_quantize.py
"""
from pathlib import Path

from ultralytics import YOLO

STUDENT_WEIGHTS = "runs/detect/runs/detect/yolo26n_imgsz1280-7/weights/best.pt"
DATA_YAML = "dataset/data.yaml"
IMGSZ = 1280


def evaluate(weights_path: str, label: str) -> None:
    model = YOLO(weights_path)
    metrics = model.val(data=DATA_YAML, imgsz=IMGSZ)
    print(f"[{label}] mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}  "
          f"speed={metrics.speed['inference']:.2f}ms/img")


def main() -> None:
    print("=== 1) FP32 (원본, 비교 기준) ===")
    evaluate(STUDENT_WEIGHTS, "FP32")

    print("\n=== 2) FP16 TensorRT export ===")
    model = YOLO(STUDENT_WEIGHTS)
    fp16_path = model.export(format="engine", half=True, imgsz=IMGSZ)
    evaluate(fp16_path, "FP16 TensorRT")

    print("\n=== 3) INT8 TensorRT export (calibration 필요) ===")
    model = YOLO(STUDENT_WEIGHTS)
    int8_path = model.export(format="engine", int8=True, data=DATA_YAML, imgsz=IMGSZ)
    evaluate(int8_path, "INT8 TensorRT")

    print("\n특히 quad_civil(소형 객체) 클래스의 mAP가 INT8에서 얼마나 떨어지는지")
    print("확인하려면, evaluate() 호출 시 verbose=True를 켜서 클래스별 결과를 보세요.")


if __name__ == "__main__":
    main()