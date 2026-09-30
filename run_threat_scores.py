"""
HybridSort 추적 결과 전체 시퀀스에 대해 threat_score.py의 위협 점수를 일괄 계산하고
CSV로 정리한다. 점수/등급 분포가 상식적으로 나오는지 확인하는 용도.

사용법:
    python run_threat_scores.py

주의: threat_score.py가 같은 폴더에 있어야 import가 됩니다.
"""
import csv
from pathlib import Path

from threat_score import (
    compute_approach_score,
    compute_loiter_score,
    compute_straightness_score,
    load_mot_txt,
    threat_level,
)

TRACKER_NAME = "hybridsort"
RESULTS_ROOT = Path("trackeval_data/trackers/mot_challenge/DroneVal") / TRACKER_NAME / "data"
OUTPUT_CSV = Path("threat_scores.csv")


def main() -> None:
    rows = []
    for mot_txt in sorted(RESULTS_ROOT.glob("*.txt")):
        seq_name = mot_txt.stem
        tracks = load_mot_txt(mot_txt)
        for tid, positions in tracks.items():
            approach = compute_approach_score(positions)
            loiter = compute_loiter_score(positions)
            straight = compute_straightness_score(positions)
            score = 0.5 * approach + 0.3 * loiter + 0.2 * straight
            level = threat_level(score)
            rows.append(
                {
                    "sequence": seq_name,
                    "track_id": tid,
                    "num_frames": len(positions),
                    "approach": round(approach, 3),
                    "loiter": round(loiter, 3),
                    "straight": round(straight, 3),
                    "score": round(score, 3),
                    "level": level,
                }
            )

    if not rows:
        print(f"{RESULTS_ROOT}에서 결과를 찾지 못했습니다.")
        return

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    high = sum(1 for r in rows if r["level"] == "고위협")
    mid = sum(1 for r in rows if r["level"] == "중위협")
    low = sum(1 for r in rows if r["level"] == "저위협")

    print(f"{len(rows)}개 track 처리 완료 -> {OUTPUT_CSV}")
    print(f"고위협 {high} / 중위협 {mid} / 저위협 {low}")


if __name__ == "__main__":
    main()