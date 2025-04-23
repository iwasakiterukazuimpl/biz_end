import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models

# モデルのロード
model = torch.load("full_model.pth", weights_only=False)
model.eval()

# クラス名（適宜変更）
class_names = ["燃えるごみ", "プラスチック", "ペットボトル", "段ボール"]

# 画像前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])



# タイトル
st.title("ごみ分別AI（PyTorchモデル）")
st.write("画像をアップロードして、種類を分類してみましょう！")

# 画像アップロード
uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    # st.image(image, caption="アップロード画像", use_column_width=True)
    st.image(image, caption="アップロード画像", use_container_width=True)

    # 前処理して推論
    input_tensor = transform(image).unsqueeze(0)  # バッチ次元を追加

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        label = class_names[predicted.item()]

    st.success(f"これは **{label}** です！")