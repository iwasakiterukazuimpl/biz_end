import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# クラスラベル（学習時のフォルダ名と一致）
class_names = ["0_burnable", "1_plastic", "2_pet", "3_cardboard"]

# 画像ファイルのパス（←ここを書き換える！）
img_path = "project_folder/testdata/testdata_4.jpeg"

# モデルを読み込む
model = tf.keras.models.load_model("gomi_model_.h5")

# 画像を読み込んで整形（学習時と同じサイズ）
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0  # 正規化

# 予測
predictions = model.predict(img_array)
predicted_index = np.argmax(predictions[0])
predicted_label = class_names[predicted_index]
confidence = predictions[0][predicted_index] * 100

# 結果を表示
print(f"予測されたラベル: {predicted_label}")
print(f"信頼度: {confidence:.2f}%")
