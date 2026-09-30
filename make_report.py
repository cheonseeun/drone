"""
TrackEval 평가 결과 + YOLO26n mAP50을 합쳐 팀원 스타일의 비교 표를 만든다.

사전 조건:
- README.md에 안내된 대로 TrackEval을 실행해서
  TrackEval/data/trackers/mot_challenge/DroneVal/<tracker>/pedestrian_summary.txt
  파일들이 이미 생성되어 있어야 한다.

사용법:
    python make_report.py
"""
from pathlib import Path

import pandas as pd

TRACKERS = ["bytetrack", "botsort", "ocsort", "hybridsort"]
TRACKEVAL_ROOT = Path("TrackEval/data/trackers/mot_challenge/DroneVal-val")

# train_yolo26.py 실행 후 출력된 mAP50 값을 여기에 직접 넣는다.
DETECTOR_MAP50 = 0.6370


def load_summary(tracker_name: str) -> dict:
    summary_file = TRACKEVAL_ROOT / tracker_name / "pedestrian_summary.txt"
    if not summary_file.exists():
        raise FileNotFoundError(
            f"{summary_file}가 없습니다. README.md의 TrackEval 실행 단계를 먼저 완료하세요."
        )
    with open(summary_file) as f:
        header = f.readline().split()
        values = f.readline().split()
    return dict(zip(header, values))


def main() -> None:
    rows = []
    for name in TRACKERS:
        metrics = load_summary(name)
        rows.append(
            {
                "tracker": name,
                "mAP50": DETECTOR_MAP50,
                "HOTA": round(float(metrics["HOTA"]), 4),
                "IDF1": round(float(metrics["IDF1"]), 4),
                "ID_switches": int(float(metrics["IDSW"])),
            }
        )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv("tracker_comparison.csv", index=False)
    print("\ntracker_comparison.csv 로 저장했습니다.")


if __name__ == "__main__":
    main()