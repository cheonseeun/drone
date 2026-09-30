\# 소형 드론 실시간 탐지 및 위협 분류 — 매의눈



2026 항공·드론 산업수요 기반 문제 해결형 해커톤 — 과제5: 소형 드론 탐지



실시간 영상에서 소형 무인기를 탐지·추적하고, 이동 궤적을 분석해 위협 수준을 판단한 뒤

Jetson급 엣지 장비에 배포 가능한 경량 파이프라인으로 최적화하는 프로젝트입니다.



\## 파이프라인 개요



```

탐지(Detection) → 추적(Tracking) → 위협 판단(Threat Scoring) → 경량화(Quantization)

&#x20;  YOLO26n            HybridSort         규칙 기반 점수화         TensorRT FP16

```



각 단계는 독립적으로 실행 가능하며, 아래 "스크립트 구성"에서 단계별 진입점을 안내합니다.



\## 데이터셋



\*\*이 저장소에는 데이터셋이 포함되어 있지 않습니다.\*\* 라이선스상 재배포가 금지되어 있어,

데이터가 필요한 스크립트는 대회 측에서 제공한 데이터를 별도로 받아 `dataset/` 폴더에

직접 배치해야 정상 동작합니다.



\- 출처: AIRBILITY UAV Detection Dataset (Hackathon Edition), 에어빌리티 제공,

&#x20; 2026 항공·드론 산업수요 기반 해커톤

\- 구성: 27,008 프레임 (train 21,383 / val 5,625), 70개 시퀀스, 1280x720 JPEG

\- 클래스: `0: quad\_civil`(민간 쿼드콥터형), `1: fixed\_wing`(고정익), `2: target\_uav`(표적 UAV)

\- 원본 라벨은 6필드(`class cx cy w h track\_id`)이며, ultralytics 학습을 위해

&#x20; `prepare\_labels.py`로 track\_id를 제거한 5필드로 변환해서 사용합니다.

\- 추적 정답(GT)은 MOTChallenge 포맷(`mot/{split}/<sequence>/gt/gt.txt`)으로 별도 제공됩니다.

\- 데이터셋 자체는 언리얼 기반 렌더링으로 생성된 합성 데이터로, 실제 카메라 노이즈,

&#x20; 렌즈 왜곡 등은 반영되어 있지 않습니다.



\## 실행 환경



\- Windows, conda 가상환경

\- NVIDIA GPU (개발 시 RTX 3050, 6GB VRAM 기준)

\- Python 3.10

\- 주요 패키지 버전은 `requirements.txt` 참고 (ultralytics 8.4.162, torch 2.11.0+cu128 기준)



```cmd

conda create -n drone python=3.10 -y

conda activate drone

pip install -r requirements.txt

```



\## 스크립트 구성



\### 1단계 — 데이터 준비

| 스크립트 | 설명 |

|---|---|

| `prepare\_labels.py` | 6필드 라벨(class cx cy w h track\_id)을 ultralytics 학습용 5필드로 변환. 원본은 자동 백업 |



\### 2단계 — Detection 모델 학습 및 비교

| 스크립트 | 설명 |

|---|---|

| `train\_yolo26.py` | YOLO26n을 640/1280 두 해상도로 학습 (최종 채택 모델) |

| `train\_yolov8\_p2.py` | YOLOv8n에 P2(stride 4) 헤드를 추가해 소형 객체 탐지 개선 여부 검증 |

| `train\_teacher\_yolo26s.py` | Knowledge Distillation용 teacher(YOLO26s) 학습 |

| `train\_rtdetr.py` | RT-DETR-l 비교 실험 (수렴 속도·정확도 열세로 조기 중단) |



\### 3단계 — Tracking 통합 및 비교

| 스크립트 | 설명 |

|---|---|

| `run\_tracking.py` | 학습된 detector와 ByteTrack, BotSort, OcSort, HybridSort 4종을 검증 시퀀스에 적용, MOT 포맷으로 결과 저장 |

| `prepare\_trackeval.py` | 추적 결과와 GT를 TrackEval 폴더 구조로 변환 (클래스는 단일 클래스로 통일해 tracker 성능만 비교) |

| `make\_report.py` | TrackEval 결과(HOTA/IDF1/ID switches)와 mAP50을 합쳐 tracker 비교표 생성 |

| `visualize\_tracking.py` | 특정 tracker의 결과를 원본 프레임에 그려 mp4로 저장 |

| `visualize\_compare.py` | 4개 tracker 결과를 한 화면에 2x2로 동시 비교하는 mp4 생성 |



\### 4단계 — 위협 판단

| 스크립트 | 설명 |

|---|---|

| `threat\_score.py` | 추적 궤적으로부터 규칙 기반 위협 점수 계산 (bbox 면적 증가율 기반 접근 판단, 체공·선회 지속시간, 경로 직선성) |

| `run\_threat\_scores.py` | 전체 시퀀스에 위협 점수를 일괄 적용해 CSV로 정리 |



\### 5단계 — 경량화

| 스크립트 | 설명 |

|---|---|

| `export\_quantize.py` | 최종 detector를 FP32/FP16/INT8로 각각 TensorRT export하고 정확도·속도 비교 |



\### 부가

| 스크립트 | 설명 |

|---|---|

| `detector\_comparison.py` | 여러 detection 실험 결과(mAP50, GFLOPs, 클래스별 성능)를 표로 정리 |

| `check\_domain\_gap.py` | 실사 검증 데이터와 별도 환경(예: 시뮬레이터) 간 탐지 결과 차이를 비교해 도메인 갭 여부 확인 |

| `make\_model\_card.py` | 학습된 가중치(.pt)에서 학습 설정과 성능을 자동 추출해 모델 카드(md) 생성. 팀원 간 모델 공유 시 imgsz 등 설정 혼동 방지용 |



\## 주요 결과



\### Detection 모델 비교 (val 기준)



| 모델 | 입력 해상도 | mAP50 | mAP50-95 | quad\_civil mAP50 |

|---|---|---|---|---|

| YOLO26n | 640 | 0.502 | 0.304 | 0.243 |

| YOLO26n | 1280 | 0.650 | 0.415 | 0.418 |

| YOLOv8n-P2 | 640 | 0.509 | 0.305 | 0.272 |

| YOLO26s | 640 | 0.535 | 0.335 | 0.284 |

| RT-DETR-l | 640 | 0.326 (best, epoch 24) | - | - |



소형 객체(quad\_civil) 탐지력은 모델 용량 증대(n에서 s)나 구조 변경(P2 헤드)보다

입력 해상도 확보(640에서 1280)가 압도적으로 효과적이었습니다. 이에 따라 Knowledge

Distillation을 통한 소형 모델 성능 개선은 이번 데이터셋에서 큰 이득이 없다고 판단해

최종 파이프라인에서 제외했습니다.



\### Tracker 비교 (YOLO26n 1280 기준, 14개 검증 시퀀스 합산)



| Tracker | HOTA | MOTA | IDF1 | ID switches |

|---|---|---|---|---|

| ByteTrack | 45.45 | 42.84 | 57.72 | 91 |

| BotSort | 40.41 | 35.81 | 47.81 | 91 |

| OcSort | 48.80 | 46.05 | 62.28 | 41 |

| HybridSort | 56.70 | 56.38 | 73.12 | 11 |



HybridSort가 전 지표에서 우위를 보여 최종 tracker로 채택했습니다.



\### 양자화 비교 (YOLO26n 1280 기준)



| 정밀도 | mAP50 | 추론 속도 | quad\_civil mAP50 |

|---|---|---|---|

| FP32 | 0.650 | 10.64ms | 0.418 |

| FP16 | 0.643 | 5.89ms | 0.420 |

| INT8 | 0.607 | 5.41ms | 0.373 |



FP16이 정확도 손실을 사실상 무시할 수준으로 유지하면서 속도를 약 1.8배 개선해

최종 배포 정밀도로 채택했습니다. INT8은 FP16 대비 속도 이득(약 5%)에 비해

소형 객체 정확도 손실(quad\_civil 약 -11%)이 커서 채택하지 않았습니다.



참고: 위 양자화 결과는 개발 GPU(RTX 3050)에서 검증한 것으로, TensorRT 엔진은

빌드한 하드웨어에 종속적입니다. 실제 Jetson 배포 시에는 해당 장비에서 export를

다시 수행해야 하며, 이번 실험은 FP16을 채택한다는 방법론적 결론을 제공합니다.



\## 알려진 한계



\- 도메인 갭: 실사 기반 학습 데이터와 시뮬레이터(예: Unity) 렌더링 환경 사이에

&#x20; 텍스처와 조명 통계 차이로 인한 오탐지가 관찰되었습니다. 실제 카메라 환경에서

&#x20; 재검증이 필요합니다.

\- quad\_civil 클래스: 세 클래스 중 객체 크기가 가장 작아 지속적으로 가장 낮은

&#x20; 탐지 성능을 보였습니다. 1280 해상도에서도 recall 0.396 수준에 머뭅니다.

\- 위협 점수 산식: 초기 규칙 기반 버전으로, 임계값과 가중치는 실제 궤적 데이터

&#x20; 분포에 맞춰 추가 보정이 필요합니다.



\## 데이터셋 출처 및 라이선스



학습 데이터: AIRBILITY UAV Detection Dataset (Hackathon Edition), 에어빌리티 제공,

2026 항공·드론 산업수요 기반 해커톤



데이터셋은 대회 규정에 따라 재배포가 금지되어 있어 이 저장소에 포함하지 않았습니다.

