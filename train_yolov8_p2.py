"""
YOLOv8n-P2 학습 스크립트 — P2(stride 4) 탐지 헤드가 추가된 버전.

목적: 1280으로 해상도를 올리는 대신, 640 입력에서 P2 헤드만 추가해도
quad_civil(소형 객체) 탐지가 얼마나 개선되는지 확인한다.
연산량은 1280의 1/4 수준이라, 효과가 비슷하다면 Jetson 배포에 훨씬 유리하다.

사용법:
    python train_yolov8_p2.py
"""
from ultralytics import YOLO


def train(imgsz: int, epochs: int = 100, batch: int = 16, data: str = "dataset/data.yaml"):
    # P2는 사전학습된 .pt 체크포인트가 아니라 구조(yaml)로 제공된다.
    # yolov8n.pt의 backbone 가중치를 가져와서 전이학습한다.
    model = YOLO("yolov8-p2.yaml")
    model.load("yolov8n.pt")

    model.train(
        data=data,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        patience=30,
        project="runs/detect",
        name=f"yolov8n_p2_imgsz{imgsz}",
        val=True,
    )

    metrics = model.val()
    print(f"[imgsz={imgsz}] mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")

    weights_path = f"runs/detect/yolov8n_p2_imgsz{imgsz}/weights/best.pt"
    return weights_path, metrics.box.map50


if __name__ == "__main__":
    # YOLO26n(640)과 같은 조건(640, 해당 배치)에서 비교
    weights_640, map50_640 = train(imgsz=640, batch=16)

    print("\n=== YOLOv8n-P2(640) 결과 ===")
    print(f"640: mAP50={map50_640:.4f}  weights={weights_640}")
    print("\n비교 기준:")
    print("  YOLO26n  640  mAP50=0.5022")
    print("  YOLO26n 1280  mAP50=0.6497")