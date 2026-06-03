# Image_segmentation-MRI-

---

## 📋 목차

1. [데이터 전처리 및 개선 사항](#1-데이터-전처리-및-개선-사항)
2. [모델 구조 개선](#2-모델-구조-개선)
3. [학습 전략 및 하이퍼파라미터](#3-학습-전략-및-하이퍼파라미터)
4. [성능 분석 및 평가](#4-성능-분석-및-평가)

---

## 1. 데이터 전처리 및 개선 사항

### Baseline vs 개선 비교

| 항목 | Baseline | 개선 |
|------|----------|------|
| 리사이징 | 단순 Resize (256×256) | 동일 |
| 정규화 | 픽셀 정규화 (÷255 → float32) | 동일 |
| 마스크 | 이진화 (>127) | 동일 |
| 데이터 증강 | 없음 | **3종 증강 전략 추가** |
| 검증 셋 분리 | validation_split=0.1만 | **train_test_split으로 90/10 명시 분리** |
| 학습 epoch | 5 epoch 고정 | **EarlyStopping + 최대 60 epoch** |
| Shuffle | 없음 | **매 에포크마다 Data Shuffle 적용** |

---

### 데이터 정규화 및 기초 전처리

#### 입력 데이터 표준화
- **해상도 최적화** — 원본 이미지를 모델 연산 효율을 고려하여 256×256 크기로 리사이징
- **픽셀 정규화** — 0~255 정수 데이터를 0.0~1.0 float32로 변환하여 학습 안정성 확보
- **마스크 이진화** — Label을 127 기준 0과 1로 이진화하여 세그멘테이션 경계 학습 명확화

#### 학습 안정성 확보
- **Data Shuffle** — 매 에포크마다 학습 순서를 랜덤으로 섞어 데이터 순서 의존성 제거
- **Type Casting** — 메모리 효율을 위해 float32 스펙으로 통일
- **Valid Split (10%)** — 전체 데이터의 10%를 검증 셋으로 명시 분리하여 과적합 실시간 모니터링

---

### 데이터 증강(Augmentation) 고도화

3가지 전략을 새롭게 적용하여 다양한 학습 환경 시뮬레이션

#### 1. Geometric Transform
- 좌우/상하 반전 (p=0.5)
- 90도 단위 무작위 회전 (k=0,1,2,3)
- 이미지·마스크 동일 변환 적용
- 카메라 각도 변화에 대한 강건성 확보

#### 2. Color Jittering
- 밝기(Brightness) ±15%
- 대비(Contrast) 0.8~1.2배
- 채도(Saturation) 0.8~1.2배
- 색조(Hue) ±5% 랜덤 조절

#### 3. 입력 정규화 클리핑
- Color Jitter 적용 후 0.0~1.0 범위로 클리핑
- 색상 왜곡으로 인한 범위 이탈 방지 처리

---

## 2. 모델 구조 개선

### Base U-Net 대비 개선 포인트

| 구성 요소 | 개선 내용 |
|-----------|-----------|
| **Encoder — ResNet50V2** (전이 학습) | 단순 Conv 블록 → Residual Connection 포함 ResNet50V2로 교체<br>ImageNet 사전 학습 가중치로 초반 수렴 속도 및 최종 정확도 개선<br>Pre-activation 구조 (BN→ReLU→Conv)로 기울기 소실 문제 해결 |
| **Attention — CBAM** (집중도 강화) | 채널 주의 집중: 피처 맵에서 중요 채널에 가중치 부여 (ratio=8)<br>공간 주의 집중: 객체 위치에 집중하여 세그멘테이션 정밀도 향상<br>Up-sampling 과정에서 Attention 모듈로 손실된 경계 정보 복원 |
| **Decoder — Skip-Connection** (디테일 복원) | 인코더 저수준 특징을 디코더에 직접 전달<br>해상도 복원 시 발생하는 디테일 손실 최소화<br>경계선 및 세밀한 세그멘테이션 마스크 품질 향상 |

---

### 개선 모델 아키텍처 구조도
INPUT          ENCODER (ResNet50V2)              BOTTLENECK       DECODER + CBAM          OUTPUT
Image    →   E1(64ch/256²)                                    D3(64ch/256²)    →    Mask
256×256×3    E2(128ch/128²)   →   Bottleneck   →   D2(128ch/128²)                 256×256×1
E3(256ch/64²)        (512ch/32²)       D1(256ch/64²)
↓                                      ↑  ↑  ↑
Skip-Connection  ─────────────────────────────────
CBAM  CBAM  CBAM
---

## 3. 학습 전략 및 하이퍼파라미터

| 항목 | 설정값 | 선정 근거 |
|------|--------|-----------|
| **Optimizer** | Adam (Weight Decay 1e-5) | 학습 초반 빠른 수렴과 모멘텀 관리, Weight Decay로 과적합 방지 |
| **Learning Rate** | Warmup (5ep) + Cosine Decay (m_mul=0.85) | 초반 5 epoch 선형 증가로 가중치 폭주 방지 → 코사인 감소로 정밀 튜닝 |
| **Loss Function** | Combined Loss (×4) | BCE + Dice + Focal + Tversky 각 0.25 결합 → 클래스 불균형·경계 불명확성 동시 해결 |
| **Batch Size** | 8 (Gradient Accumulation) | GPU 메모리 내 최대화 및 데이터 증강 효과 극대화 |
| **Epochs** | 최대 60 (EarlyStopping p=15) | val_dice_coefficient 기준 15 epoch 비개선 시 조기 종료 |
| **TTA** | 8방향 앙상블 (추론 단계) | 원본·반전·회전·전치 8가지 변환 예측값 가중 평균 → 예측 변동성 최소화 |

> **Combined Loss 공식:**
> ```
> L = 0.25·L_BCE + 0.25·L_Dice + 0.25·L_Focal + 0.25·L_Tversky
> ```

---

## 4. 성능 분석 및 평가

### 학습 곡선 분석

- Train/Val Loss 모두 감소하며 **42 epoch에서 EarlyStopping 발동**, 두 곡선 간 gap이 크지 않아 과적합 없이 안정적으로 수렴
- Dice Coefficient 기준으로도 Val이 Train을 거의 추종하며 상승 → Data Augmentation과 Weight Decay(1e-5)의 정규화 효과
- Warmup LR이 초반 5 epoch 동안 가중치 폭주를 방지, 이후 Cosine Decay가 수렴 후반부의 미세 조정을 담당

---

### 단계별 성능 기여도 분석 (Ablation Study)

| 단계 | Dice (%) | IoU (%) | 기여 |
|------|----------|---------|------|
| Baseline (U-Net) | 52 | 41 | — |
| + ResNet50V2 전이학습 | 68 | 54 | **+16%p** |
| + CBAM + Combined Loss | 76 | 63 | **+8%p** |
| + Augmentation + Warmup LR | 81 | 71 | **+5%p** |
| + TTA (최종) | **86** | **79** | **+5.4%p** |

**핵심 기여 요인:**
- **ResNet50V2 (+16% Dice)** — 사전 학습된 필터로 특징 추출 품질이 크게 향상. 학습 데이터가 상대적으로 적은 경우 전이학습 효과가 증폭되는 경향이 본 실험에서도 동일하게 나타남
- **CBAM + Combined Loss (+8% Dice)** — 경계 영역 집중 및 False Negative 패널티 강화 효과
- **TTA (+5.4% Dice)** — 추가 학습 없이 8방향 앙상블로 예측 분산을 줄여 비용 대비 효율이 높은 전략

---

### 최종 성능 지표

| 지표 | Baseline | **최종 (Improved)** | 향상폭 |
|------|----------|---------------------|--------|
| **Dice Score** | 52% | **86.4%** | 🔺 +34.4%p |
| **IoU Score** | 41% | **79.2%** | 🔺 +38.2%p |

**성능 향상 요인 종합:**
1. **Pretrained ResNet50V2 백본** — 사전 학습 필터로 특징 추출 레이어의 수렴이 정밀해짐
2. **TTA 8방향 앙상블** — 테스트 단계에서 예측값 평균으로 변동성 최소화
3. **Tversky Loss** — False Negative 패널티 강화로 Recall 수치 대폭 상승

> Baseline 대비 Dice +34.4%p, IoU +38.2%p의 향상은 단일 요인이 아닌 **전 구성요소의 복합적 시너지 결과**임.
> ResNet50V2 백본으로 특징 추출 품질을 확보하고, CBAM으로 경계 영역 집중력을 높였으며, Combined Loss의 Tversky 항이 False Negative에 강한 패널티를 부여해 Recall을 끌어올렸음. 여기에 Augmentation으로 다양한 조건에 대한 일반화 능력을 갖추고, TTA로 추론 변동성을 최소화한 것이 최종 지표에 고르게 반영됨.
