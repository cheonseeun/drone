"""
학습된 YOLO26n 탐지기 + 4가지 tracker(ByteTrack, BotSort, OcSort, HybridSort) 비교 실행

각 tracker로 검증용 영상들을 돌려서 결과를 MOT 포맷(txt)으로 저장한다.
저장된 결과는 이후 TrackEval로 HOTA / IDF1 / ID switch를 계산하는 데 쓰인다.

주의: boxmot 25.0.0 기준, create_tracker는 TrackerSpec dataclass를 요구해서
      다루기 까다롭습니다. 대신 dir(boxmot)로 확인된 개별 tracker 클래스
      (ByteTrack, BotSort, OcSort, HybridSort)를 실제 생성자 시그니처에 맞춰
      직접 생성하는 방식을 씁니다. 탐지는 ultralytics YOLO를 직접 사용합니다.

사용법:
    python run_tracking.py

주의:
- 시퀀스(영상)마다 tracker 인스턴스를 새로 만들어야 한다. 그렇지 않으면
  track id가 영상 간에 이어져서(carry over) 평가가 틀어진다.
- DETECTOR_WEIGHTS는 train_yolo26.py에서 나온 best.pt 경로로 바꿔서 사용한다.
- REID_WEIGHTS는 botsort/hybridsort처럼 외형 특징을 쓰는 tracker에 필요하며,
  최초 실행 시 자동으로 다운로드된다 (인터넷 연결 필요).
"""
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from boxmot import BotSort, ByteTrack, HybridSort, OcSort

DETECTOR_WEIGHTS = "runs/detect/runs/detect/yolo26n_imgsz1280-7/weights/best.pt"
SEQUENCES_DIR = "dataset/images/val"      # <시퀀스 폴더>/<6자리 프레임>.jpg 구조
OUTPUT_ROOT = Path("trackeval_data/trackers/mot_challenge/DroneVal")
REID_WEIGHTS = Path("osnet_x0_25_msmt17.pt")  # botsort/hybridsort용 ReID 모델 (자동 다운로드)
DEVICE = "cuda:0"
FRAME_RATE = 30

# tracker_name -> 새 인스턴스를 만드는 함수 (시퀀스마다 새로 호출해서 ID를 리셋한다)
TRACKER_FACTORIES = {
    "bytetrack": lambda: ByteTrack(frame_rate=FRAME_RATE),
    "botsort": lambda: BotSort(
        reid_weights=REID_WEIGHTS, device=DEVICE, half=False, frame_rate=FRAME_RATE
    ),
    "ocsort": lambda: OcSort(),
    "hybridsort": lambda: HybridSort(
        reid_weights=REID_WEIGHTS, device=DEVICE, half=False
    ),
}
TRACKERS = list(TRACKER_FACTORIES.keys())


def to_boxmot_dets(result) -> np.ndarray:
    """ultralytics 결과 -> boxmot이 요구하는 (N, 6) [x1,y1,x2,y2,conf,cls] 배열로 변환"""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 6))
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy().reshape(-1, 1)
    cls = boxes.cls.cpu().numpy().reshape(-1, 1)
    return np.hstack([xyxy, conf, cls])


def run_one_sequence(seq_dir: Path, tracker_name: str, detector: YOLO, out_dir: Path) -> None:
    tracker = TRACKER_FACTORIES[tracker_name]()  # 시퀀스마다 새로 생성 (ID 리셋 목적)

    frame_paths = sorted(seq_dir.glob("*.jpg"))
    lines = []

    for frame_idx, frame_path in enumerate(frame_paths, start=1):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"  경고: {frame_path} 를 읽지 못했습니다. 건너뜁니다.")
            continue

        result = detector.predict(frame, imgsz=1280, verbose=False)[0]
        dets = to_boxmot_dets(result)
        tracks = tracker.update(dets, frame)  # (M, 8): x1,y1,x2,y2,id,conf,cls,det_ind

        for t in tracks:
            x1, y1, x2, y2, tid, conf = t[0], t[1], t[2], t[3], t[4], t[5]
            w, h = x2 - x1, y2 - y1
            # MOT Challenge 포맷: frame,id,bb_left,bb_top,bb_w,bb_h,conf,-1,-1,-1
            lines.append(
                f"{frame_idx},{int(tid)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1"
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    seq_name = seq_dir.name
    (out_dir / f"{seq_name}.txt").write_text("\n".join(lines))
    print(f"  [{tracker_name}] {seq_name}: {len(frame_paths)} frames 처리 완료")


def main() -> None:
    detector = YOLO(DETECTOR_WEIGHTS)
    sequence_dirs = sorted(p for p in Path(SEQUENCES_DIR).iterdir() if p.is_dir())

    if not sequence_dirs:
        raise FileNotFoundError(f"{SEQUENCES_DIR} 안에서 시퀀스 폴더를 찾지 못했습니다.")

    for tracker_name in TRACKERS:
        print(f"=== {tracker_name} 실행 ===")
        out_dir = OUTPUT_ROOT / tracker_name / "data"
        for seq_dir in sequence_dirs:
            run_one_sequence(seq_dir, tracker_name, detector, out_dir)


if __name__ == "__main__":
    main()