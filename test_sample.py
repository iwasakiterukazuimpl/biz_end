import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# モデル読み込み
model = tf.keras.models.load_model('gomi_model.h5')  # .kerasでもOK！

# クラス名（順番はimage_dataset_from_directoryで作った時と同じ）
class_names = ['0_burnable', '1_plastic', '2_pet', '3_cardboard']

# 推論したい画像のパス
img_path = 'testdata/testdata_6.jpeg'  # ← 自分の画像に変更してください

# 画像読み込み & 前処理（モデルの入力サイズに合わせる）
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # バッチ次元追加
img_array = img_array / 255.0  # 正規化（もし学習時にしていれば）

# 推論
predictions = model.predict(img_array)

# 結果表示
print("予測スコア:", predictions[0])  # 各クラスの確率的なスコア
predicted_index = np.argmax(predictions[0])
print("予測クラス:", predicted_index)
print("予測クラス名:", class_names[predicted_index])

# 画像も表示してみる
plt.imshow(img)
plt.title(f"Predicted: {class_names[predicted_index]}")
plt.axis("off")
plt.show()
