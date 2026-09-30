"""
원본(실사) val 이미지에 대한 탐지 결과를 뽑아서, 유니티 렌더링 환경에서의
오탐지·낮은 confidence 문제가 도메인 갭 때문인지 확인한다.

사용법:
    python check_domain_gap.py
"""
from ultralytics import YOLO

WEIGHTS = "runs/detect/runs/detect/yolo26n_imgsz1280-7/weights/best.pt"  # 실제 가중치 경로로 바꿔서 사용
SOURCE = "dataset/images/val/val_001"
IMGSZ = 1280
CONF = 0.25


def main() -> None:
    model = YOLO(WEIGHTS)
    model.predict(source=SOURCE, imgsz=IMGSZ, conf=CONF, save=True)
    print("완료. runs/detect/predict 폴더에서 결과 이미지를 확인하세요.")


if __name__ == "__main__":
    main()