"""
RT-DETR 드론 탐지 모델 학습 스크립트 (train_yolo26.py와 동일한 구조, 모델만 교체)

제안서에 적었던 "YOLO와 RT-DETR 비교" 실험을 위한 스크립트.
RT-DETR은 트랜스포머 기반, NMS-free 구조로 YOLO26과 접근 자체가 다르다.

사용법:
    python train_rtdetr.py

주의:
- RT-DETR은 attention 연산 때문에 YOLO보다 VRAM을 더 많이 씁니다.
  6GB(RTX 3050)에서는 batch를 낮춰서 시작하는 걸 권장합니다.
"""
from ultralytics import RTDETR


def train(imgsz: int, epochs: int = 100, batch: int = 4, data: str = "dataset/data.yaml"):
    """지정 해상도로 RT-DETR을 처음부터 학습하고 검증 mAP를 반환한다."""
    model = RTDETR("rtdetr-l.pt")

    model.train(
        data=data,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        patience=30,
        project="runs/detect",
        name=f"rtdetr_imgsz{imgsz}",
        val=True,
    )

    metrics = model.val()
    print(f"[imgsz={imgsz}] mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")

    weights_path = f"runs/detect/rtdetr_imgsz{imgsz}/weights/best.pt"
    return weights_path, metrics.box.map50


if __name__ == "__main__":
    # 640으로 먼저 베이스라인 확보 (YOLO26n과 동일한 조건으로 비교하기 위해)
    weights_640, map50_640 = train(imgsz=640, batch=4)

    print("\n=== RT-DETR 결과 ===")
    print(f"640: mAP50={map50_640:.4f}  weights={weights_640}")
    print("\nYOLO26n(640) mAP50=0.5022 와 비교해보세요.")