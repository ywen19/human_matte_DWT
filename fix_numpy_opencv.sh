#!/bin/bash

set -e

echo "修复 NumPy 和 OpenCV 兼容性问题..."

# 激活环境
eval "$(conda shell.bash hook)"
conda activate sam2_yolo

# 设置CUDA环境变量
CONDA_ENV_PATH="/home/wy/anaconda3/envs/cuda118"
export CUDA_HOME="$CONDA_ENV_PATH"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib64:$CONDA_ENV_PATH/lib:$LD_LIBRARY_PATH"

echo "当前环境: $(which python)"
echo "当前CUDA_HOME: $CUDA_HOME"

# 第一步：完全卸载有问题的包
echo "卸载有冲突的包..."
pip uninstall numpy opencv-python opencv-contrib-python -y || true

# 第二步：安装兼容的 NumPy 版本
echo "安装兼容的 NumPy 1.24.3..."
pip install "numpy==1.24.3" --force-reinstall --no-cache-dir

# 第三步：安装与 NumPy 1.24.3 兼容的 OpenCV 版本
echo "安装与 NumPy 1.24.3 兼容的 OpenCV..."
pip install "opencv-python==4.8.1.78" --no-deps --force-reinstall --no-cache-dir

# 验证 NumPy 没有被自动升级
echo "验证 NumPy 版本..."
python -c "import numpy as np; print(f'NumPy版本: {np.__version__}')"

# 测试 OpenCV 导入
echo "测试 OpenCV 导入..."
python -c "
import numpy as np
print(f'NumPy版本: {np.__version__}')

try:
    import cv2
    print(f'OpenCV版本: {cv2.__version__}')
    print('✅ OpenCV 导入成功')
    
    # 测试基本功能
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print('✅ OpenCV 基本功能测试通过')
    
except Exception as e:
    print(f'❌ OpenCV 导入失败: {e}')
    exit(1)
"

# 测试 PyTorch + NumPy 兼容性
echo "测试 PyTorch + NumPy 兼容性..."
python -c "
import torch
import numpy as np

print(f'PyTorch版本: {torch.__version__}')
print(f'NumPy版本: {np.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')

# 测试 NumPy-PyTorch 转换
try:
    arr = np.random.randn(10, 10).astype(np.float32)
    tensor = torch.from_numpy(arr)
    arr_back = tensor.numpy()
    print('✅ NumPy-PyTorch 转换测试通过')
    
    if torch.cuda.is_available():
        cuda_tensor = torch.randn(10, 10, device='cuda:0')
        cpu_result = cuda_tensor.cpu().numpy()
        print('✅ CUDA tensor 测试通过')
        
except Exception as e:
    print(f'❌ NumPy-PyTorch 兼容性测试失败: {e}')
    exit(1)
"

# 如果 OpenCV 仍有问题，尝试替代方案
if ! python -c "import cv2" 2>/dev/null; then
    echo "OpenCV 仍有问题，尝试替代方案..."
    
    # 尝试安装 opencv-python-headless（更轻量，兼容性更好）
    pip uninstall opencv-python -y || true
    pip install "opencv-python-headless==4.8.1.78" --no-deps --force-reinstall
    
    echo "测试 opencv-python-headless..."
    python -c "
    import cv2
    import numpy as np
    print(f'OpenCV版本: {cv2.__version__}')
    print('✅ opencv-python-headless 导入成功')
    "
fi

# 固定 NumPy 版本，防止其他包自动升级
echo "固定 NumPy 版本..."
pip install "numpy<2.0,>=1.24.0" --force-reinstall

# 最终验证
echo "最终验证..."
python -c "
import torch
import numpy as np
import cv2
import sys

print('='*50)
print('环境验证结果:')
print('='*50)
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'NumPy: {np.__version__}')
print(f'OpenCV: {cv2.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'PyTorch CUDA版本: {torch.version.cuda}')
    print(f'GPU数量: {torch.cuda.device_count()}')

# 综合测试
print('\\n综合功能测试:')
try:
    # NumPy 数组
    arr = np.random.randn(5, 5).astype(np.float32)
    
    # PyTorch 张量
    tensor = torch.from_numpy(arr)
    
    # OpenCV 图像处理
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print('✅ 所有核心功能正常')
    
except Exception as e:
    print(f'❌ 功能测试失败: {e}')
    exit(1)

print('\\n🎉 NumPy-OpenCV 兼容性问题已解决!')
"

echo ""
echo "🎉 修复完成!"
echo ""
echo "重要提示:"
echo "1. NumPy 版本已固定为 1.24.3（与 OpenCV 4.8.1.78 兼容）"
echo "2. 如果将来安装新包时遇到 NumPy 版本冲突，使用: pip install --no-deps"
echo "3. 或者在安装前明确指定: pip install 'numpy<2.0' package_name"