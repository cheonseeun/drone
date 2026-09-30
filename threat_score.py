"""
추적 결과(MOT 포맷)로부터 규칙 기반 위협 점수를 계산한다.

입력: run_tracking.py로 만든 tracker 출력 txt
      (frame,id,bb_left,bb_top,bb_w,bb_h,conf,-1,-1,-1)
출력: track id별 위협 점수(0~1)와 등급(저/중/고위협)

기준 3가지 (제안서와 동일하되, 접근 속도는 카메라 자체 움직임에 맞게 재정의):
  1. bbox 면적 증가율 (카메라 쪽으로 접근할수록 화면 속 크기가 커진다는 원리 -
     U60 카메라 자체가 호버링/순항으로 계속 움직이므로, 화면 속 고정 좌표까지의
     거리보다 이 방식이 카메라 자체 움직임에 훨씬 덜 민감하다)
  2. 특정 지점에서의 체공·선회 지속 시간
  3. 이동 경로의 직선성 여부

주의: 아래 임계값(THRESH_*, LOITER_*, GROWTH 스케일)은 초기 규칙 기반 버전을 위한
      임의값이며, 실제 궤적 데이터가 쌓이면 데이터 기반으로 보정해야 한다.

사용법:
    python threat_score.py trackeval_data/trackers/mot_challenge/DroneVal/bytetrack/data/seq01.txt
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

FPS = 30  # 영상 fps

LOITER_RADIUS = 50       # 이 반경(px) 안에서 머무르면 "체공"으로 간주
LOITER_MIN_FRAMES = 60   # 60프레임(2초 @30fps) 이상 머물러야 체공으로 판정

THRESH_HIGH = 0.7
THRESH_MID = 0.4


def load_mot_txt(path: Path):
    """track id별로 (frame, center_x, center_y, area) 리스트를 만든다."""
    tracks = defaultdict(list)
    for line in path.read_text().strip().splitlines():
        if not line.strip():
            continue
        frame, tid, x, y, w, h, conf, *_ = line.split(",")
        w, h = float(w), float(h)
        cx = float(x) + w / 2
        cy = float(y) + h / 2
        area = w * h
        tracks[int(tid)].append((int(frame), cx, cy, area))
    for tid in tracks:
        tracks[tid].sort(key=lambda r: r[0])
    return tracks


def compute_approach_score(positions) -> float:
    """bbox 면적 증가율 -> 0~1. 화면 속 크기가 빠르게 커질수록 접근으로 판단한다."""
    if len(positions) < 2:
        return 0.0
    start_area = max(positions[0][3], 1.0)
    end_area = positions[-1][3]
    frames_elapsed = max(positions[-1][0] - positions[0][0], 1)
    # 시작 크기 대비 상대 증가율(스케일 불변) / 경과 프레임
    relative_growth_rate = (end_area - start_area) / start_area / frames_elapsed
    return max(0.0, min(1.0, relative_growth_rate * 200.0))  # 스케일은 초기값, 추후 보정 대상


def compute_loiter_score(positions) -> float:
    """특정 지점 체공·선회 지속 시간 -> 0~1"""
    max_loiter = 0
    window_start = 0
    for i in range(1, len(positions)):
        _, cx0, cy0, _ = positions[window_start]
        _, cx1, cy1, _ = positions[i]
        if math.hypot(cx1 - cx0, cy1 - cy0) > LOITER_RADIUS:
            window_start = i  # 반경을 벗어나면 새 구간 시작
        else:
            max_loiter = max(max_loiter, positions[i][0] - positions[window_start][0])
    return min(1.0, max_loiter / (LOITER_MIN_FRAMES * 3))


def compute_straightness_score(positions) -> float:
    """이동 경로의 직선성 -> 0~1 (1에 가까울수록 직선 이동)"""
    if len(positions) < 2:
        return 0.0
    _, x0, y0, _ = positions[0]
    _, x1, y1, _ = positions[-1]
    straight_dist = math.hypot(x1 - x0, y1 - y0)
    path_len = sum(
        math.hypot(positions[i][1] - positions[i - 1][1], positions[i][2] - positions[i - 1][2])
        for i in range(1, len(positions))
    )
    return straight_dist / path_len if path_len > 0 else 0.0


def threat_level(score: float) -> str:
    if score >= THRESH_HIGH:
        return "고위협"
    if score >= THRESH_MID:
        return "중위협"
    return "저위협"


def main(mot_txt_path: str) -> None:
    tracks = load_mot_txt(Path(mot_txt_path))
    for tid, positions in tracks.items():
        approach = compute_approach_score(positions)
        loiter = compute_loiter_score(positions)
        straight = compute_straightness_score(positions)

        # 가중합 (초기값, 추후 실제 궤적 데이터로 가중치·임계값 보정 대상)
        score = 0.5 * approach + 0.3 * loiter + 0.2 * straight
        level = threat_level(score)

        print(
            f"track {tid}: approach={approach:.2f} loiter={loiter:.2f} "
            f"straight={straight:.2f} -> score={score:.2f} ({level})"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("사용법: python threat_score.py <MOT포맷 tracker 출력 txt>")
    main(sys.argv[1])