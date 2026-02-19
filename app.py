import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# --- 1. 配置与设备 ---
st.set_page_config(page_title="Medical AI Diagnosis", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# --- 2. 核心功能：Grad-CAM (可解释性引擎) ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册钩子 (Hooks) 抓取中间层数据
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx):
        # 1. 前向传播
        self.model.zero_grad()
        output = self.model(x)
        score = output[0][class_idx]
        
        # 2. 反向传播
        score.backward()
        
        # 3. 生成热力图
        gradients = self.gradients
        activations = self.activations
        b, k, u, v = gradients.size()
        
        # 全局平均池化计算权重 (Alpha)
        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)
        
        # 权重与特征图加权求和
        cam = (weights * activations).sum(1, keepdim=True)
        
        # ReLU + 归一化
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7) # 防止除零
        
        return cam.squeeze().cpu().detach().numpy()

# --- 3. 加载模型 ---
@st.cache_resource
def load_model():
    # 必须与训练时的结构完全一致
    model = models.resnet50(weights=None) # 推理时不需要下载 ImageNet 权重
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3) # 3分类
    
    # 加载我们要训练好的权重
    try:
        model.load_state_dict(torch.load('medical_resnet.pth', map_location=device))
        model.to(device)
        model.eval() # 开启评估模式
        return model
    except FileNotFoundError:
        st.error("❌ 找不到模型文件 medical_resnet.pth，请先运行 train_model.py")
        return None

model = load_model()
target_layer = model.layer4[2].conv3 # ResNet50 的最后一个卷积层
grad_cam = GradCAM(model, target_layer)

# --- 4. 图像预处理 ---
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

classes = ['bacterial', 'normal', 'viral'] # 注意顺序要和训练时一致（字母序）

# --- 5. 界面 UI ---
st.title("🩻 AI-Assisted Medical Imaging Diagnosis")
st.markdown("**Core Tech:** ResNet50 | Transfer Learning | Grad-CAM Interpretability")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload X-Ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file and model:
    # 加载图片
    image = Image.open(uploaded_file).convert('RGB')
    
    # 显示原图
    with col1:
        st.image(image, caption="Original X-Ray", use_column_width=True)
        st.info("🔍 Analyzing...")

    # 推理
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # 获取预测结果
    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    prediction_idx = torch.argmax(probabilities).item()
    prediction_label = classes[prediction_idx]
    confidence = probabilities[prediction_idx].item()

    # 生成 Grad-CAM 热力图
    heatmap = grad_cam(input_tensor, prediction_idx)
    
    # 将热力图叠加到原图上
    # 1. 调整热力图大小以匹配原图
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_img = heatmap_img.resize(image.size, resample=Image.BICUBIC)
    
    # 2. 上色 (使用 matplotlib 的 jet 颜色映射)
    cmap = plt.get_cmap('jet')
    heatmap_colored = cmap(np.array(heatmap_img) / 255.0) # 归一化后上色
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8) # 去掉 alpha 通道
    heatmap_pil = Image.fromarray(heatmap_colored)
    
    # 3. 混合
    overlayed_img = Image.blend(image, heatmap_pil, alpha=0.4)

    # 结果展示
    with col2:
        st.image(overlayed_img, caption=f"Grad-CAM Heatmap (AI Focus Area)", use_column_width=True)
        
        st.subheader("Diagnostic Result:")
        if prediction_label == "normal":
            st.success(f"✅ **Prediction: Normal** ({confidence:.1%})")
        else:
            st.error(f"⚠️ **Prediction: {prediction_label.capitalize()} Pneumonia** ({confidence:.1%})")
            
        st.markdown("""
        **Interpretation:**
        The **red/yellow** regions in the image on the right indicate the areas that most influenced the AI's decision.
        """)