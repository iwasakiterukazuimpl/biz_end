import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# パラメータ
img_height = 224
img_width = 224
batch_size = 32
data_dir = "dataset"

# データ拡張の設定（より強力な拡張を追加）
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomBrightness(0.3),
    tf.keras.layers.RandomContrast(0.3),
    tf.keras.layers.RandomTranslation(0.2, 0.2),
])

# データの読み込み
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print("クラスの順番:", class_names)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# クラスの重みを計算
y_train = []
for _, labels in train_ds:
    y_train.extend(labels.numpy())
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print("\nクラスの重み:", class_weight_dict)

# パフォーマンス最適化
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Normalization レイヤーの統計情報を計算
normalization_layer = tf.keras.layers.Normalization()
for images, _ in train_ds.take(100):
    normalization_layer.adapt(images)

# データ拡張を適用
train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=AUTOTUNE
)

# EfficientNetB0モデル
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)

# ファインチューニングの設定
trainable_layers = int(len(base_model.layers) * 0.3)  # 30%の層を訓練可能に
for layer in base_model.layers[:-trainable_layers]:
    layer.trainable = False
for layer in base_model.layers[-trainable_layers:]:
    layer.trainable = True

# モデルの構築
model = models.Sequential([
    # 前処理層
    tf.keras.layers.Rescaling(1./255),
    normalization_layer,
    
    # ベースモデル
    base_model,
    
    # 追加の層
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Dense(512, activation='swish', kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Dense(256, activation='swish', kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    # 最終層の活性化関数をsoftmaxからsigmoidに変更
    tf.keras.layers.Dense(len(class_names), activation='sigmoid')
])

# 学習率スケジューラの設定
initial_learning_rate = 0.0001
decay_steps = 1000
decay_rate = 0.9
learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps, decay_rate
)

# コールバックの設定
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6
)

# モデルのコンパイル
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.Precision(),
             tf.keras.metrics.Recall(),
             tf.keras.metrics.AUC()]
)

# クラスごとの予測分布を表示する関数
def plot_prediction_distribution(history, class_names):
    plt.figure(figsize=(12, 6))
    for metric in ['precision', 'recall']:
        plt.subplot(1, 2, 1 if metric == 'precision' else 2)
        for i, class_name in enumerate(class_names):
            values = history.history[f'{metric}_{i}'] if f'{metric}_{i}' in history.history else []
            if values:
                plt.plot(values, label=f'{class_name} {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} per Class')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()

# モデルの学習
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weight_dict
)

# 精度のグラフ
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# クラスごとの予測分布を表示
plot_prediction_distribution(history, class_names)

# 精度の表示
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
print(f"\n最終精度: 訓練={acc[-1]:.4f}, 検証={val_acc[-1]:.4f}")

# 各クラスの最終的な性能指標を表示
print("\n各クラスの最終性能:")
for i, class_name in enumerate(class_names):
    precision = history.history[f'precision_{i}'][-1] if f'precision_{i}' in history.history else 0
    recall = history.history[f'recall_{i}'][-1] if f'recall_{i}' in history.history else 0
    print(f"{class_name}:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")

# モデルを保存
model.save("efficientnet_model.h5")