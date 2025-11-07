

from PIL import Image
import torch, requests
from transformers import AutoImageProcessor, AutoModelForObjectDetection

# 1. 载入模型
processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# 2. 读图
url = "https://raw.githubusercontent.com/microsoft/table-transformer/main/test_data/PMC5754337_table_1.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# 3. 前处理
inputs = processor(images=image, return_tensors="pt").to(device)

# 4. 推理
with torch.no_grad():
    outputs = model(**inputs)

# 5. 后处理 → 直接拿到 boxes, scores, labels
target_sizes = torch.tensor([image.size[::-1]])  # (h, w)
results = processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=target_sizes)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    print(f"{model.config.id2label[label.item()]}: "
          f"{score:.3f}  {box.round().tolist()}")