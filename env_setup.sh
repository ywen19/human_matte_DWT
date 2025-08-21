#!/bin/bash

set -e

echo "配置SAM2 + YOLO环境（基于Conda CUDA 11.8）..."

# 检查cuda118环境是否存在
if ! conda env list | grep -q "cuda118"; then
    echo "❌ cuda118环境未找到，请先运行 conda_cuda_install.sh"
    exit 1
fi

# 激活cuda118环境获取CUDA路径
eval "$(conda shell.bash hook)"
conda activate cuda118
CONDA_ENV_PATH=$(conda info --envs | grep "cuda118" | head -1 | awk '{print $NF}')

if [ -z "$CONDA_ENV_PATH" ]; then
    CONDA_ENV_PATH="$CONDA_PREFIX"
fi

if [ -z "$CONDA_ENV_PATH" ]; then
    CONDA_ENV_PATH="/home/wy/anaconda3/envs/cuda118"
fi

CUDA_HOME="$CONDA_ENV_PATH"
echo "使用CUDA环境: $CUDA_HOME"

# 设置CUDA环境变量
export CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH"

# 验证CUDA库
echo "验证CUDA库..."
if [ -f "$CUDA_HOME/lib/libcudart.so" ]; then
    echo "✅ 找到CUDA运行时库"
    ls -la "$CUDA_HOME/lib/libcudart.so"*
else
    echo "⚠️ 未找到标准CUDA库，但继续安装..."
fi

# 删除现有环境（可选）
read -p "是否删除现有的sam2_yolo环境? (y/N): " remove_env
if [[ $remove_env =~ ^[Yy]$ ]]; then
    conda env remove -n sam2_yolo -y || true
fi

# 创建基础环境（不安装pip依赖）
echo "创建sam2_yolo基础环境..."
conda create -n sam2_yolo python=3.10 pip=23.0 ffmpeg git -y

# 激活新环境
conda activate sam2_yolo

# 重新设置CUDA环境变量（在新环境中）
export CUDA_HOME="$CONDA_ENV_PATH"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib64:$CONDA_ENV_PATH/lib:$LD_LIBRARY_PATH"

# 配置pip镜像源以加速下载
echo "配置pip镜像源..."
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.timeout 300
pip config set global.retries 5

# 确保卸载任何现有的PyTorch（防止CPU版本残留）
echo "清理现有PyTorch安装..."
pip uninstall torch torchvision torchaudio -y || true

# 安装PyTorch GPU版本（使用CUDA 11.8兼容版本）
echo "安装PyTorch 2.0.1 + CUDA 11.8 GPU版本..."
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118 \
    --force-reinstall --no-cache-dir --timeout 300

# 验证确实安装了GPU版本
echo "验证PyTorch GPU安装..."
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
if '+cu118' in torch.__version__:
    print('✅ 确认安装了CUDA 11.8 GPU版本')
elif 'cpu' in torch.__version__:
    print('❌ 错误！安装了CPU版本，退出脚本')
    exit(1)
else:
    print(f'⚠️ 版本字符串异常: {torch.__version__}')
    
print(f'CUDA编译支持: {torch.version.cuda}')
print(f'CUDA运行时可用: {torch.cuda.is_available()}')
"

if [ $? -ne 0 ]; then
    echo "❌ PyTorch GPU版本验证失败，停止安装"
    exit 1
fi

# 分批安装核心依赖，避免网络超时
echo "分批安装核心依赖..."

# 第一批：基础科学计算库
echo "安装基础科学计算库..."
pip install --timeout 300 --retries 5 \
    "numpy>=1.21.0,<2.0.0" \
    "scipy>=1.9.0,<2.0.0" \
    "matplotlib>=3.7.0,<4.0.0" \
    "pillow>=9.0.0,<11.0.0"

# 第二批：计算机视觉库
echo "安装计算机视觉库..."
pip install --timeout 300 --retries 5 \
    "opencv-python>=4.8.0,<5.0.0" \
    "imageio>=2.30.0,<3.0.0" \
    "av>=0.5.0,<12.0.0"

# 第三批：深度学习和工具库
echo "安装深度学习和工具库..."
pip install --timeout 300 --retries 5 \
    "transformers>=4.30.0,<5.0.0" \
    "huggingface_hub>=0.20.0" \
    "ultralytics>=8.0.0,<9.0.0" \
    "kornia>=0.7.0,<1.0.0"

# 第四批：其他依赖
echo "安装其他依赖..."
pip install --timeout 300 --retries 5 \
    "pycocotools>=2.0.0" \
    "jupyterlab" \
    "ipywidgets" \
    "hydra-core>=1.3.0" \
    "iopath>=0.1.0" \
    "cython" \
    "gitpython>=3.0.0" \
    "hickle>=5.0.0" \
    "tensorboard>=2.10.0" \
    "tqdm>=4.60.0" \
    "gradio>=3.30.0,<5.0.0" \
    "gdown>=4.0.0" \
    "einops>=0.6.0,<1.0.0" \
    "PySide6>=6.0.0" \
    "charset-normalizer>=3.0.0" \
    "netifaces>=0.10.0" \
    "cchardet>=2.0.0" \
    "easydict" \
    "PyWavelets>=1.4.0,<2.0.0" \
    "setuptools>=60.0.0" \
    "wheel>=0.40.0"

# 确保numpy版本兼容
echo "安装兼容的numpy版本..."
pip install numpy==1.24.3 --force-reinstall

# 重新安装opencv确保兼容性
echo "重新安装opencv确保兼容性..."
pip install opencv-python==4.8.1.78 --force-reinstall

# 验证安装
echo "验证完整安装..."
python -c "
import torch
import numpy as np
import cv2
import os

print(f'Python: {__import__(\"sys\").version}')
print(f'PyTorch: {torch.__version__}')
print(f'NumPy: {np.__version__}')
print(f'OpenCV: {cv2.__version__}')
print(f'CUDA_HOME: {os.environ.get(\"CUDA_HOME\", \"未设置\")}')
print(f'CUDA可用: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'PyTorch CUDA版本: {torch.version.cuda}')
    print(f'GPU数量: {torch.cuda.device_count()}')
    for i in range(min(torch.cuda.device_count(), 8)):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')

# 测试numpy-torch兼容性
print('\\n测试numpy-torch兼容性...')
arr = np.random.randn(10, 10).astype(np.float32)
tensor = torch.from_numpy(arr)
print('✅ NumPy-PyTorch转换测试通过')

# 测试CUDA tensor操作
if torch.cuda.is_available():
    cuda_tensor = torch.randn(10, 10, device='cuda:0')
    cpu_result = cuda_tensor.cpu().numpy()
    print('✅ CUDA tensor操作测试通过')
else:
    print('⚠️  CUDA不可用，请检查安装')
"

# 检查SAM2
if [ -d "sam2" ]; then
    echo "发现已存在的SAM2目录..."
    read -p "是否重新安装SAM2? (y/N): " reinstall_sam2
    if [[ $reinstall_sam2 =~ ^[Yy]$ ]]; then
        echo "重新安装SAM2..."
        cd sam2
        pip install -e .
        cd ..
    fi
else
    echo "克隆SAM2仓库..."
    git clone https://github.com/facebookresearch/sam2.git
    cd sam2
    pip install -e .
    cd ..
fi

# 测试YOLO
echo "测试YOLO..."
python -c "
from ultralytics import YOLO
import numpy as np
import torch

print('初始化YOLO模型...')
model = YOLO('yolov8n.pt')
test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

print('运行YOLO推理...')
results = model(test_img, verbose=False)
print('✅ YOLO CPU测试通过')

# 测试GPU推理
if torch.cuda.is_available():
    print('测试YOLO GPU推理...')
    results_gpu = model(test_img, device='cuda:0', verbose=False)
    print('✅ YOLO GPU测试通过')
"

echo ""
echo "🎉 环境配置完成!"
echo ""
echo "使用方法:"
echo "1. 激活环境: conda activate sam2_yolo"
echo "2. 设置CUDA环境变量:"
echo "   export CUDA_HOME=$CONDA_ENV_PATH"
echo "   export PATH=$CONDA_ENV_PATH/bin:\$PATH" 
echo "   export LD_LIBRARY_PATH=$CONDA_ENV_PATH/lib64:$CONDA_ENV_PATH/lib:\$LD_LIBRARY_PATH"
echo ""
echo "建议将环境变量添加到 ~/.bashrc 以便永久使用"
echo ""
echo "版本信息:"
echo "- PyTorch: 2.0.1+cu118"
echo "- CUDA: 11.8 (通过Conda)"
echo "- GPU支持: 8x RTX 3090"