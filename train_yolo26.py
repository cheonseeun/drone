"""
YOLO26n 드론 탐지 모델 학습 스크립트

사용법:
    python train_yolo26.py

drone.yaml 예시 (같은 폴더에 두세요):
    path: ./dataset
    train: images/train
    val: images/val
    names:
      0: drone
"""
from ultralytics import YOLO


def train(imgsz: int, epochs: int = 100, batch: int = 8, data: str = "dataset/data.yaml"):
    """지정 해상도로 YOLO26n을 처음부터 학습하고 검증 mAP를 반환한다."""
    model = YOLO("yolo26n.pt")

    model.train(
        data=data,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        patience=30,          # 30 epoch 동안 개선 없으면 조기 종료
        project="runs/detect",
        name=f"yolo26n_imgsz{imgsz}",
        val=True,
    )

    metrics = model.val()
    print(f"[imgsz={imgsz}] mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")

    weights_path = f"runs/detect/yolo26n_imgsz{imgsz}/weights/best.pt"
    return weights_path, metrics.box.map50


if __name__ == "__main__":
    # Jetson 실시간성을 고려해 640으로 낮춘 버전도 함께 확보
    weights_640, map50_640 = train(imgsz=640, batch=16)
    # 팀원 방식과 동일하게: 먼저 고해상도(1280)로 학습
    weights_1280, map50_1280 = train(imgsz=1280, batch=8)


    print("\n=== 해상도별 비교 ===")
    print(f"1280: mAP50={map50_1280:.4f}  weights={weights_1280}")
    print(f" 640: mAP50={map50_640:.4f}  weights={weights_640}")
    print("\n다음 단계(run_tracking.py)에서 사용할 weights 경로를 정하세요.")