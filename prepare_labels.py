"""
YOLO 라벨의 6번째 컬럼(track_id)을 제거해 ultralytics 학습용 5컬럼 포맷으로 변환한다.

원본 라벨 포맷 (data.yaml 주석 참고):
    class cx cy w h track_id
변환 후:
    class cx cy w h

track_id는 dataset/mot/{split}/<seq>/gt.txt의 id 컬럼과 동일하므로,
tracking GT가 필요할 때는 mot 폴더 쪽을 그대로 쓰면 된다.
(이 스크립트는 dataset/labels를 dataset/labels_with_track_id로 백업한 뒤
 dataset/labels 안의 파일들을 5컬럼으로 덮어쓴다.)

사용법 (drone 폴더에서 실행):
    python prepare_labels.py --labels_dir dataset/labels
"""
import argparse
import shutil
from pathlib import Path


def strip_track_id(labels_dir: Path) -> None:
    backup_dir = labels_dir.parent / f"{labels_dir.name}_with_track_id"
    if not backup_dir.exists():
        print(f"백업 생성: {labels_dir} -> {backup_dir}")
        shutil.copytree(labels_dir, backup_dir)
    else:
        print(f"백업이 이미 존재합니다: {backup_dir} (건너뜀)")

    txt_files = list(labels_dir.rglob("*.txt"))
    print(f"{len(txt_files)}개 라벨 파일을 5컬럼으로 변환합니다...")

    bad_count = 0
    for txt_path in txt_files:
        lines = txt_path.read_text().strip().splitlines()
        new_lines = []
        for line in lines:
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) == 6:
                fields = fields[:5]  # track_id 제거
            elif len(fields) != 5:
                bad_count += 1
                print(f"경고: 예상치 못한 컬럼 수({len(fields)}) - {txt_path}")
                continue
            new_lines.append(" ".join(fields))
        txt_path.write_text("\n".join(new_lines))

    print(f"완료. (형식 이상 파일 {bad_count}개)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_dir", default="dataset/labels")
    args = parser.parse_args()
    strip_track_id(Path(args.labels_dir))