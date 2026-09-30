"""
run_tracking.py가 만든 MOT 포맷 결과를 원본 프레임 위에 그려서 mp4 영상으로 저장한다.
tracker별로 결과가 눈으로 어떻게 다른지 비교할 때 쓴다.

사용법:
    python visualize_tracking.py --tracker bytetrack --seq val_001
    python visualize_tracking.py --tracker botsort --seq val_001 --fps 15

여러 시퀀스를 한 번에 뽑고 싶으면 --seq 없이 실행 (해당 tracker의 전체 시퀀스 처리):
    python visualize_tracking.py --tracker bytetrack
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

FRAMES_ROOT = Path("dataset/images/val")
TRACKER_RESULTS_ROOT = Path("trackeval_data/trackers/mot_challenge/DroneVal")
OUTPUT_DIR = Path("viz_out")


def load_tracks_by_frame(mot_txt: Path) -> dict:
    """frame_idx -> [(track_id, x1, y1, x2, y2, conf), ...]"""
    by_frame = {}
    for line in mot_txt.read_text().strip().splitlines():
        if not line.strip():
            continue
        frame, tid, x, y, w, h, conf, *_ = line.split(",")
        frame = int(frame)
        x1, y1 = float(x), float(y)
        x2, y2 = x1 + float(w), y1 + float(h)
        by_frame.setdefault(frame, []).append((int(tid), x1, y1, x2, y2, float(conf)))
    return by_frame


def color_for_id(track_id: int) -> tuple:
    """track_id마다 고정된 색을 배정 (동일 ID는 항상 같은 색)"""
    rng = np.random.default_rng(track_id * 9973 + 1)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


def render_sequence(seq_name: str, tracker_name: str, fps: int) -> None:
    frame_dir = FRAMES_ROOT / seq_name
    mot_txt = TRACKER_RESULTS_ROOT / tracker_name / "data" / f"{seq_name}.txt"
    if not mot_txt.exists():
        print(f"건너뜀: {mot_txt} 가 없습니다.")
        return

    tracks_by_frame = load_tracks_by_frame(mot_txt)
    frame_paths = sorted(frame_dir.glob("*.jpg"))
    if not frame_paths:
        print(f"건너뜀: {frame_dir}에 프레임이 없습니다.")
        return

    first = cv2.imread(str(frame_paths[0]))
    h, w = first.shape[:2]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{tracker_name}_{seq_name}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for frame_idx, frame_path in enumerate(frame_paths, start=1):
        frame = cv2.imread(str(frame_path))
        for tid, x1, y1, x2, y2, conf in tracks_by_frame.get(frame_idx, []):
            color = color_for_id(tid)
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(frame, p1, p2, color, 2)
            label = f"ID {tid} ({conf:.2f})"
            cv2.putText(
                frame, label, (p1[0], max(p1[1] - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
            )
        writer.write(frame)

    writer.release()
    print(f"저장 완료: {out_path} ({len(frame_paths)} frames)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", required=True, choices=["bytetrack", "botsort", "ocsort", "hybridsort"])
    parser.add_argument("--seq", default=None, help="특정 시퀀스만 처리 (예: val_001). 생략 시 전체 시퀀스.")
    parser.add_argument("--fps", type=int, default=30, help="출력 영상 fps (원본 촬영 fps와 다르게 재생 속도 조절 가능)")
    args = parser.parse_args()

    if args.seq:
        render_sequence(args.seq, args.tracker, args.fps)
    else:
        seq_names = sorted(p.name for p in FRAMES_ROOT.iterdir() if p.is_dir())
        for seq_name in seq_names:
            render_sequence(seq_name, args.tracker, args.fps)


if __name__ == "__main__":
    main()