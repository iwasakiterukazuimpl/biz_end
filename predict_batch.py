import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# クラス名（学習時と一致）
class_names = ["0_burnable", "1_plastic", "2_pet", "3_cardboard"]

# モデルの読み込み
model = tf.keras.models.load_model("gomi_model.h5")

# 画像フォルダのパス
image_folder = "testdata"  # ←自分のフォルダ名に変更OK

# 各画像を処理
for filename in os.listdir(image_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(image_folder, filename)

        # 画像読み込み＆整形（MobileNetV2は224×224）
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # 予測
        predictions = model.predict(img_array, verbose=0)
        predicted_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_index] * 100
        predicted_label = class_names[predicted_index]

        print(f"{filename} → {predicted_label}（信頼度: {confidence:.2f}％）")
