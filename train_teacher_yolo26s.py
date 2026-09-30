"""
Knowledge Distillation의 teacher로 쓸 YOLO26s를 우리 데이터로 fine-tuning한다.

distillation은 teacher와 student가 같은 입력을 보고 feature를 비교하는 방식이므로,
teacher가 COCO만 알고 있으면 (우리 드론 데이터를 모르면) 가르쳐줄 게 없다.
그래서 student(YOLO26n)와 똑같이, teacher(YOLO26s)도 우리 데이터로 먼저 학습시킨다.

사용법:
    python train_teacher_yolo26s.py
"""
from ultralytics import YOLO


def train(imgsz: int = 640, epochs: int = 100, batch: int = 8, data: str = "dataset/data.yaml"):
    model = YOLO("yolo26s.pt")

    model.train(
        data=data,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        patience=30,
        project="runs/detect",
        name=f"yolo26s_teacher_imgsz{imgsz}",
        val=True,
    )

    metrics = model.val()
    print(f"[Teacher YOLO26s, imgsz={imgsz}] mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")

    weights_path = f"runs/detect/yolo26s_teacher_imgsz{imgsz}/weights/best.pt"
    return weights_path, metrics.box.map50


if __name__ == "__main__":
    # 640 기준 (student도 640에서 배포할 계획이므로 같은 해상도로 맞춘다)
    # RTX 3050(6GB) 기준 YOLO26n(batch=16)보다 모델이 커서 batch를 낮춰서 시작
    weights_path, map50 = train(imgsz=640, batch=8)

    print("\n=== Teacher(YOLO26s, 640) 결과 ===")
    print(f"mAP50={map50:.4f}  weights={weights_path}")
    print("\n비교 기준:")
    print("  YOLO26n(student, distillation 전)  640  mAP50=0.5022")
    print("  이 teacher가 이보다 확실히 높아야 distillation 의미가 있습니다.")