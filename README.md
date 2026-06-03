# Image_segmentation-MRI-
> **범례:** ResNet50V2 Encoder | Decoder | CBAM Attention | Skip-Connection

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
