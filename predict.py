# mini_predict.py
import torch
from torchvision import transforms, models
from PIL import Image
import os

# クラスラベル（ImageFolderで読み込んだ順）
class_names = ['burnable', 'plastic', 'pet', 'cardboard']

# モデルを再構築して読み込む
model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 4)
model.load_state_dict(torch.load('model.pth'))
model.eval()
 
# 推論用前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 推論対象のディレクトリ
test_dir = 'testdata'
image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# 各画像に対して推論を実行
for image_file in image_files:
    image_path = os.path.join(test_dir, image_file)
    try:
        # 画像を読み込んで推論
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)

        # 推論実行
        with torch.no_grad():
            outputs = model(img_tensor)
            predicted = outputs.argmax(1).item()
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()

        print(f"📸 画像: {image_file}")
        print(f"🧠 予測: {class_names[predicted]} (確信度: {confidence:.2%})")
        print("-" * 50)
    except Exception as e:
        print(f"❌ エラー ({image_file}): {str(e)}")
        print("-" * 50)
