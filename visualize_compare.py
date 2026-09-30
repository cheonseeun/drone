"""
하나의 시퀀스에 대해 4개 tracker(ByteTrack, BotSort, OcSort, HybridSort) 결과를
동시에 그려서 2x2 화면 분할 영상 하나로 합친다.

사용법:
    python visualize_compare.py --seq val_013
    python visualize_compare.py --seq val_013 --fps 10
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

FRAMES_ROOT = Path("dataset/images/val")
TRACKER_RESULTS_ROOT = Path("trackeval_data/trackers/mot_challenge/DroneVal")
OUTPUT_DIR = Path("viz_compare_out")

TRACKERS = ["bytetrack", "botsort", "ocsort", "hybridsort"]
LABEL_COLORS = {
    "bytetrack": (255, 200, 0),
    "botsort": (0, 200, 255),
    "ocsort": (0, 255, 120),
    "hybridsort": (255, 80, 80),
}


def load_tracks_by_frame(mot_txt: Path) -> dict:
    """frame_idx -> [(track_id, x1, y1, x2, y2, conf), ...]"""
    by_frame = {}
    if not mot_txt.exists():
        return by_frame
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
    rng = np.random.default_rng(track_id * 9973 + 1)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


def draw_panel(base_frame: np.ndarray, tracker_name: str, tracks: list) -> np.ndarray:
    frame = base_frame.copy()
    for tid, x1, y1, x2, y2, conf in tracks:
        color = color_for_id(tid)
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, color, 2)
        cv2.putText(
            frame, f"ID {tid}", (p1[0], max(p1[1] - 6, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
        )
    # 좌상단에 tracker 이름 라벨 (배경 박스 + 텍스트)
    label = tracker_name.upper()
    label_color = LABEL_COLORS[tracker_name]
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (0, 0), (tw + 16, th + 16), (0, 0, 0), -1)
    cv2.putText(frame, label, (8, th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2, cv2.LINE_AA)
    return frame


def render_comparison(seq_name: str, fps: int) -> None:
    frame_dir = FRAMES_ROOT / seq_name
    frame_paths = sorted(frame_dir.glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"{frame_dir}에 프레임이 없습니다.")

    tracks_by_tracker = {
        name: load_tracks_by_frame(TRACKER_RESULTS_ROOT / name / "data" / f"{seq_name}.txt")
        for name in TRACKERS
    }

    first = cv2.imread(str(frame_paths[0]))
    h, w = first.shape[:2]
    panel_w, panel_h = w // 2, h // 2
    grid_w, grid_h = panel_w * 2, panel_h * 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"compare_{seq_name}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (grid_w, grid_h))

    for frame_idx, frame_path in enumerate(frame_paths, start=1):
        base_frame = cv2.imread(str(frame_path))
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        for i, tracker_name in enumerate(TRACKERS):
            tracks = tracks_by_tracker[tracker_name].get(frame_idx, [])
            panel = draw_panel(base_frame, tracker_name, tracks)
            panel = cv2.resize(panel, (panel_w, panel_h))

            row, col = divmod(i, 2)
            y0, x0 = row * panel_h, col * panel_w
            grid[y0:y0 + panel_h, x0:x0 + panel_w] = panel

        writer.write(grid)

    writer.release()
    print(f"저장 완료: {out_path} ({len(frame_paths)} frames, {grid_w}x{grid_h})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", required=True, help="비교할 시퀀스 (예: val_013)")
    parser.add_argument("--fps", type=int, default=30, help="출력 영상 fps")
    args = parser.parse_args()
    render_comparison(args.seq, args.fps)


if __name__ == "__main__":
    main()