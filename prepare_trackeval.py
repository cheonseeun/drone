"""
dataset/mot/val 의 GT와 run_tracking.py가 만든 tracker 결과를,
TrackEval이 기대하는 폴더 구조로 복사/변환한다.

TrackEval 기본 설정은 GT의 class 컬럼이 1(pedestrian)인 박스만 평가 대상으로 삼고
나머지 class는 방해 요소(distractor)로 취급한다. 클래스별 탐지 정확도는 이미 학습
단계(mAP)에서 확인했으므로, 여기서는 모든 GT 박스의 class를 1로 통일해서
"드론이면 다 추적 대상"으로 tracker 성능만 비교한다.

사용법 (TrackEval을 git clone 한 뒤, drone_project 폴더에서 실행):
    python prepare_trackeval.py --trackeval_root TrackEval
"""
import argparse
import shutil
from pathlib import Path

BENCHMARK = "DroneVal"
SPLIT = "val"
TRACKERS = ["bytetrack", "botsort", "ocsort", "hybridsort"]


def remap_gt_class_to_1(src_gt: Path, dst_gt: Path) -> None:
    lines = []
    for line in src_gt.read_text().strip().splitlines():
        if not line.strip():
            continue
        fields = line.split(",")
        fields[7] = "1"  # class 컬럼을 1(pedestrian 자리)로 통일
        lines.append(",".join(fields))
    dst_gt.parent.mkdir(parents=True, exist_ok=True)
    dst_gt.write_text("\n".join(lines))


def main(trackeval_root: Path) -> None:
    src_mot_root = Path("dataset/mot") / SPLIT
    seq_names = sorted(p.name for p in src_mot_root.iterdir() if p.is_dir())
    if not seq_names:
        raise FileNotFoundError(f"{src_mot_root} 안에서 시퀀스를 찾지 못했습니다.")

    gt_root = trackeval_root / "data" / "gt" / "mot_challenge" / f"{BENCHMARK}-{SPLIT}"
    seqmap_dir = trackeval_root / "data" / "gt" / "mot_challenge" / "seqmaps"
    trk_root = trackeval_root / "data" / "trackers" / "mot_challenge" / f"{BENCHMARK}-{SPLIT}"

    # 1) GT + seqinfo.ini 복사 (class는 1로 통일)
    for seq in seq_names:
        src_seq = src_mot_root / seq
        dst_seq = gt_root / seq
        remap_gt_class_to_1(src_seq / "gt" / "gt.txt", dst_seq / "gt" / "gt.txt")
        shutil.copy(src_seq / "seqinfo.ini", dst_seq / "seqinfo.ini")
        print(f"GT 복사 완료: {seq}")

    # 2) seqmap 파일 생성 (평가할 시퀀스 목록)
    seqmap_dir.mkdir(parents=True, exist_ok=True)
    seqmap_path = seqmap_dir / f"{BENCHMARK}-{SPLIT}.txt"
    seqmap_path.write_text("name\n" + "\n".join(seq_names))
    print(f"seqmap 생성 완료: {seqmap_path}")

    # 3) run_tracking.py 결과(trackeval_data/) -> TrackEval/data/trackers/ 로 복사
    src_trk_root = Path("trackeval_data/trackers/mot_challenge/DroneVal")
    for tracker_name in TRACKERS:
        src_dir = src_trk_root / tracker_name / "data"
        dst_dir = trk_root / tracker_name / "data"
        dst_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for txt_file in src_dir.glob("*.txt"):
            shutil.copy(txt_file, dst_dir / txt_file.name)
            copied += 1
        print(f"tracker 결과 복사 완료: {tracker_name} ({copied}개 시퀀스)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trackeval_root", default="TrackEval")
    args = parser.parse_args()
    main(Path(args.trackeval_root))