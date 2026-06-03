# ============================================================
# GPU 스펙 확인
# ============================================================
import subprocess
subprocess.run(["nvidia-smi"])


# ============================================================
# 라이브러리 추가
# ============================================================
import os
import random
import cv2

# Tensorflow 관련 디버그 및 경고 메시지 비활성화 (삭제 금지)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import tensorflow.image as tfi
from tensorflow.keras import Sequential
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from PIL import Image, ImageDraw
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# ============================================================
# 폴더 경로 설정
# ============================================================
data_path = '/kaggle/input/competitions/2026-DAU-CV'


# ============================================================
# 재구현 세팅
# ============================================================
def init_seeds(seed):
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    np.random.seed(seed)
    random.seed(seed)


init_seeds(2026)


# ============================================================
# 데이터 로드
# ============================================================
train_image_path = os.path.join(data_path, 'train/images')
train_label_path = os.path.join(data_path, 'train/masks')
test_image_path  = os.path.join(data_path, 'test/images')

output_path = '/kaggle/working'

train_images = os.listdir(train_image_path)
train_images = [os.path.join(train_image_path, x) for x in train_images]
train_labels = os.listdir(train_label_path)
train_labels = [os.path.join(train_label_path, x) for x in train_labels]

train_images.sort(), train_labels.sort()

test_images = os.listdir(test_image_path)
test_images = [os.path.join(test_image_path, x) for x in test_images]

test_images.sort()


# ============================================================
# 이미지 시각화
# ============================================================
sample_img   = Image.open(train_images[0]).convert("RGB")
sample_label = Image.open(train_labels[0]).convert("L")

print(f'sample image size : {sample_img.size}, sample label size : {sample_label.size}')

img_np   = np.array(sample_img)
label_np = np.array(sample_label)

label_color = cm.jet(label_np / 255.0)[:, :, :3]
overlay     = np.clip(0.7 * img_np / 255.0 + 0.3 * label_color, 0, 1)

fig = plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.imshow(img_np);     plt.title("Image");   plt.axis("off")
plt.subplot(1, 3, 2); plt.imshow(label_color); plt.title("Mask");    plt.axis("off")
plt.subplot(1, 3, 3); plt.imshow(overlay);    plt.title("Overlay"); plt.axis("off")
plt.tight_layout()
plt.show()


# ============================================================
# 데이터 전처리
# ============================================================
IMG_HEIGHT = 256
IMG_WIDTH  = 256

def build_train_dataset(image_paths, mask_paths, img_height=256, img_width=256):
    X, y = [], []
    for img_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths)):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_width, img_height))
        img = img.astype(np.float32) / 255.0
        X.append(img)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (img_width, img_height))
        mask = (mask > 127).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)
        y.append(mask)

    X = np.stack(X, axis=0)
    y = np.stack(y, axis=0)
    print(f" Dataset loaded: X={X.shape}, y={y.shape}")
    return X, y


def build_test_dataset(image_paths, img_height=256, img_width=256):
    X = []
    for img_path in tqdm(image_paths):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_width, img_height))
        img = img.astype(np.float32) / 255.0
        X.append(img)

    X = np.stack(X, axis=0)
    print(f" Test dataset loaded: X={X.shape}")
    return X


train_X, train_y = build_train_dataset(train_images, train_labels,
                                        img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
test_X = build_test_dataset(test_images, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)

# plt.imshow(train_y[0]) # 인덱스 번호를 바꾸면 다른 학습 mask 확인 가능 (필요시 주석 제거)


@tf.function
def augment(image, mask):
    # 좌우 반전
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask  = tf.image.flip_left_right(mask)
    # 상하 반전
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask  = tf.image.flip_up_down(mask)
    # 90도 단위 회전 (이미지+마스크 동일)
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    mask  = tf.image.rot90(mask,  k)
    # 밝기 / 대비 / 채도 / 색조 (이미지만)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.05)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, mask


# Cutout: 랜덤 패치를 mean값으로 덮어 일반화 향상
def apply_cutout(image, mask):
    image = image.numpy()
    h, w  = image.shape[:2]
    size  = np.random.randint(20, 48)
    cy    = np.random.randint(0, h)
    cx    = np.random.randint(0, w)
    y1, y2 = max(0, cy - size//2), min(h, cy + size//2)
    x1, x2 = max(0, cx - size//2), min(w, cx + size//2)
    image[y1:y2, x1:x2] = image.mean()
    return image, mask.numpy()


@tf.function
def augment_with_cutout(image, mask):
    image, mask = augment(image, mask)
    if tf.random.uniform(()) > 0.5:
        image, mask = tf.py_function(apply_cutout, [image, mask], [tf.float32, tf.float32])
        image.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
        mask.set_shape([IMG_HEIGHT, IMG_WIDTH, 1])
    return image, mask


BATCH_SIZE = 8

val_size = int(len(train_X) * 0.1)
val_X, val_y = train_X[-val_size:], train_y[-val_size:]
fit_X, fit_y = train_X[:-val_size], train_y[:-val_size]

fit_ds = (tf.data.Dataset.from_tensor_slices((fit_X, fit_y))
          .shuffle(buffer_size=len(fit_X), seed=2026)
          .map(augment_with_cutout, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(BATCH_SIZE)
          .prefetch(tf.data.AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((val_X, val_y))
          .batch(BATCH_SIZE)
          .prefetch(tf.data.AUTOTUNE))


# ============================================================
# 모델 정의
# ============================================================

# CBAM Attention Gate: keras.ops 사용 (Keras 3 호환)
def cbam_block(x, ratio=8):
    filters = x.shape[-1]
    # Channel Attention
    avg = layers.GlobalAveragePooling2D()(x)
    mx  = layers.GlobalMaxPooling2D()(x)
    avg = layers.Reshape((1, 1, filters))(avg)
    mx  = layers.Reshape((1, 1, filters))(mx)
    shared_dense1 = layers.Dense(filters // ratio, activation='relu', use_bias=False)
    shared_dense2 = layers.Dense(filters, use_bias=False)
    ca = layers.Activation('sigmoid')(
        layers.Add()([shared_dense2(shared_dense1(avg)),
                      shared_dense2(shared_dense1(mx))])
    )
    x = layers.Multiply()([x, ca])
    # Spatial Attention — keras.ops로 교체 (KerasTensor 호환)
    avg_sp = layers.Lambda(lambda t: tf.keras.ops.mean(t, axis=-1, keepdims=True))(x)
    max_sp = layers.Lambda(lambda t: tf.keras.ops.max(t,  axis=-1, keepdims=True))(x)
    sp = layers.Concatenate()([avg_sp, max_sp])
    sp = layers.Conv2D(1, 7, padding='same', activation='sigmoid', use_bias=False)(sp)
    x  = layers.Multiply()([x, sp])
    return x


def build_model(input_shape=(256, 256, 3)):
    # 백본: ResNet50V2 (ImageNet pretrained, keras 공식 제공)
    backbone = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
    )

    # Skip connection 레이어 이름 (ResNet50V2 기준)
    skip_names = [
        'conv1_conv',          # stride2 → 128×128  (64ch)
        'conv2_block3_1_relu', # 64×64   (64ch)
        'conv3_block4_1_relu', # 32×32   (128ch)
        'conv4_block6_1_relu', # 16×16   (256ch)
    ]
    skips  = [backbone.get_layer(n).output for n in skip_names]
    bridge = backbone.output  # 8×8 (2048ch)

    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

    def up_block(x, skip, filters):
        x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
        x = layers.Concatenate()([x, skip])
        x = conv_block(x, filters)
        x = cbam_block(x)
        return x

    # Bottleneck
    x = layers.Dropout(0.3)(bridge)
    x = conv_block(x, 512)

    # 디코더 (스킵 역순)
    x = up_block(x, skips[3], 256)
    x = up_block(x, skips[2], 128)
    x = up_block(x, skips[1], 64)
    x = up_block(x, skips[0], 32)

    # 마지막 업샘플 (원본 해상도 복원)
    x = layers.Conv2DTranspose(16, 2, strides=2, padding='same')(x)
    x = conv_block(x, 16)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)

    model = models.Model(inputs=backbone.input, outputs=outputs, name="ResNet50V2-UNet-CBAM")
    return model


# ============================================================
# 손실 함수 정의
# ============================================================

def dice_coefficient(y_true, y_pred, threshold=0.5):
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    smooth = 1e-5
    intersection = tf.reduce_sum(y_pred * y_true, axis=[1, 2, 3])
    union = tf.reduce_sum(y_pred, axis=[1, 2, 3]) + tf.reduce_sum(y_true, axis=[1, 2, 3])
    dice = (2. * intersection + smooth) / (union + smooth)
    return tf.reduce_mean(dice)


def iou_coefficient(y_true, y_pred, threshold=0.5, smooth=1e-5):
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = (tf.reduce_sum(y_true, axis=[1, 2, 3])
             + tf.reduce_sum(y_pred, axis=[1, 2, 3])
             - intersection)
    iou = (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(iou)


def dice_loss(y_true, y_pred, smooth=1e-5):
    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1.0 - tf.reduce_mean(dice)


# Focal Loss: 쉬운 픽셀 down-weight, 어려운 픽셀에 집중
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25, eps=1e-7):
    y_true  = tf.cast(y_true, tf.float32)
    y_pred  = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    bce     = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    return tf.reduce_mean(alpha_t * tf.pow(1.0 - p_t, gamma) * bce)


# Tversky Loss: FN 패널티 강화 → recall 향상 (작은 객체 세그멘테이션에 유리)
def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-5):
    y_true = tf.cast(y_true, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred,       axis=[1, 2, 3])
    fp = tf.reduce_sum((1-y_true) * y_pred,   axis=[1, 2, 3])
    fn = tf.reduce_sum(y_true * (1-y_pred),   axis=[1, 2, 3])
    return 1.0 - tf.reduce_mean((tp + smooth) / (tp + alpha*fp + beta*fn + smooth))


def combined_loss(y_true, y_pred):
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    dl  = dice_loss(y_true, y_pred)
    fl  = focal_loss(y_true, y_pred)
    tv  = tversky_loss(y_true, y_pred)
    return 0.25*bce + 0.25*dl + 0.25*fl + 0.25*tv


# ============================================================
# 모델 생성 및 컴파일
# ============================================================
model = build_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

EPOCHS           = 60
steps_per_epoch  = len(fit_X) // BATCH_SIZE
WARMUP_EPOCHS    = 5
WARMUP_STEPS     = WARMUP_EPOCHS * steps_per_epoch

# Linear Warmup → CosineDecayRestarts
# pretrained 백본 초반 보호 후 코사인 감쇠
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps, initial_lr, first_decay_steps,
                 t_mul=1.0, m_mul=0.85, alpha=1e-6):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.initial_lr   = initial_lr
        self.cosine       = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_lr,
            first_decay_steps=first_decay_steps,
            t_mul=t_mul, m_mul=m_mul, alpha=alpha,
        )

    def __call__(self, step):
        step      = tf.cast(step, tf.float32)
        warmup_lr = self.initial_lr * (step / tf.cast(self.warmup_steps, tf.float32))
        cosine_lr = self.cosine(step - self.warmup_steps)
        return tf.cond(step < self.warmup_steps, lambda: warmup_lr, lambda: cosine_lr)

    def get_config(self):
        return {'warmup_steps': self.warmup_steps, 'initial_lr': self.initial_lr}


lr_schedule = WarmupCosineDecay(
    warmup_steps      = WARMUP_STEPS,
    initial_lr        = 2e-4,                   # pretrained 백본 → 낮은 LR
    first_decay_steps = steps_per_epoch * 15,
    t_mul=1.0, m_mul=0.85, alpha=1e-6,
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, weight_decay=1e-5),
    loss=combined_loss,
    metrics=[dice_coefficient, iou_coefficient]
)

model.summary()


# ============================================================
# 모델 학습
# ============================================================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_dice_coefficient', mode='max',
        patience=15, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='/kaggle/working/best_model.keras',
        monitor='val_dice_coefficient', mode='max',
        save_best_only=True, verbose=1
    ),
    # ReduceLROnPlateau 제거 — LearningRateSchedule과 충돌
    tf.keras.callbacks.CSVLogger('/kaggle/working/training_log.csv'),
]

history = model.fit(
    fit_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# 예측값 확인 예시 (필요시 주석 제거)
# pred = model.predict(train_X[:5])
# print(pred.min(), pred.max())
# plt.hist(pred.flatten(), bins=50)
# plt.title("Predicted Probability Distribution")
# plt.xlabel("Predicted Value")
# plt.ylabel("Pixel Count")
# plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history.history['loss'],                label='Train Loss')
axes[0].plot(history.history['val_loss'],             label='Val Loss')
axes[0].set_title('Loss'); axes[0].legend()

axes[1].plot(history.history['dice_coefficient'],     label='Train Dice')
axes[1].plot(history.history['val_dice_coefficient'], label='Val Dice')
axes[1].set_title('Dice Coefficient'); axes[1].legend()

axes[2].plot(history.history['iou_coefficient'],      label='Train IoU')
axes[2].plot(history.history['val_iou_coefficient'],  label='Val IoU')
axes[2].set_title('IoU Coefficient'); axes[2].legend()

plt.tight_layout()
plt.savefig('/kaggle/working/training_curve.png', dpi=100)
plt.show()


# ============================================================
# 모델 예측
# ============================================================

def predict_with_tta(model, images, batch_size=4):
    """8가지 변환 앙상블 — 원본/단순 반전에 가중치 부여"""
    def pred(x):
        return model.predict(x, batch_size=batch_size, verbose=0)

    p0 = pred(images)                                           # 원본 (가중치 2)
    p1 = pred(images[:, :, ::-1, :])[:, :, ::-1, :]            # 좌우 반전
    p2 = pred(images[:, ::-1, :, :])[:, ::-1, :, :]            # 상하 반전
    p3 = pred(images[:, ::-1, ::-1, :])[:, ::-1, ::-1, :]      # 180도

    img_90  = np.rot90(images, k=1, axes=(1, 2))
    p4 = np.rot90(pred(img_90),  k=-1, axes=(1, 2))             # 90도

    img_270 = np.rot90(images, k=3, axes=(1, 2))
    p5 = np.rot90(pred(img_270), k=1,  axes=(1, 2))             # 270도

    img_t = np.transpose(images, (0, 2, 1, 3))
    p6 = np.transpose(pred(img_t), (0, 2, 1, 3))               # 대각 전치

    img_t2 = img_t[:, :, ::-1, :]
    p7 = np.transpose(pred(img_t2), (0, 2, 1, 3))[:, :, ::-1, :]  # 대각+좌우

    # 개선: 원본에 가중치 2 부여 (총 9로 나눔)
    return (2*p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7) / 9.0


print("TTA 예측 시작...")
pred = predict_with_tta(model, test_X, batch_size=4)
print(f"예측 완료: {pred.shape}")

# 예측결과 확인 (필요시 주석 제거)
# plt.imshow(pred[0])


# ============================================================
# Run-Length Encoding 및 예측 결과 저장
# ============================================================
output_path = '/kaggle/working'
result_path = os.path.join(output_path, 'results')
os.makedirs(result_path, exist_ok=True)


# 수정X
def rle_encode(mask):
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    if len(runs) % 2 != 0:
        runs = np.append(runs, len(pixels) - 1)
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# 수정X
THRESHOLD = 0.5

rle_list = []

for i in range(pred.shape[0]):
    prob = pred[i, :, :, 0]
    mask = (prob > THRESHOLD).astype(np.uint8)

    rle = rle_encode(mask)
    rle_list.append(rle)

    mask_img = mask * 255
    cv2.imwrite(os.path.join(result_path, f"{i+1:04d}.jpg"), mask_img)


# ============================================================
# 제출 CSV 파일 저장
# ============================================================
submission = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'))

submission['EncodedPixels'] = rle_list
print(submission.head())

submission.to_csv('sample_submission.csv', index=False)

import shutil
shutil.make_archive('label', 'zip', result_path)

shutil.rmtree(result_path)
