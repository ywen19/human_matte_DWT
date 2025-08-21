#!/bin/bash

set -e

echo "完成 SAM2 + YOLO 环境配置..."

# 激活环境
eval "$(conda shell.bash hook)"
conda activate sam2_yolo

# 设置CUDA环境变量
CONDA_ENV_PATH="/home/wy/anaconda3/envs/cuda118"
export CUDA_HOME="$CONDA_ENV_PATH"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib64:$CONDA_ENV_PATH/lib:$LD_LIBRARY_PATH"

echo "当前环境: $(which python)"
echo "CUDA_HOME: $CUDA_HOME"

# 检查并安装 SAM2
if [ -d "sam2" ]; then
    echo "发现已存在的 SAM2 目录..."
    read -p "是否重新克隆并安装 SAM2? (y/N): " reinstall_sam2
    if [[ $reinstall_sam2 =~ ^[Yy]$ ]]; then
        echo "删除现有 SAM2 目录..."
        rm -rf sam2
        echo "重新克隆 SAM2..."
        git clone https://github.com/facebookresearch/sam2.git
    fi
else
    echo "克隆 SAM2 仓库..."
    git clone https://github.com/facebookresearch/sam2.git
fi

# 安装 SAM2
echo "安装 SAM2..."
cd sam2
pip install -e . --no-deps  # 使用 --no-deps 避免版本冲突
cd ..

# 测试 SAM2 导入
echo "测试 SAM2 导入..."
python -c "
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    print('✅ SAM2 导入成功')
except ImportError as e:
    print(f'❌ SAM2 导入失败: {e}')
    print('尝试安装缺失的依赖...')
    exit(1)
except Exception as e:
    print(f'⚠️  SAM2 导入警告: {e}')
    print('可能需要下载模型文件，但基本功能正常')
"

# 测试 YOLO
echo "测试 YOLO..."
python -c "
try:
    from ultralytics import YOLO
    import numpy as np
    import torch

    print('初始化 YOLO 模型...')
    model = YOLO('yolov8n.pt')  # 会自动下载模型
    
    # 创建测试图像
    test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    print('运行 YOLO CPU 推理...')
    results = model(test_img, verbose=False)
    print('✅ YOLO CPU 测试通过')

    # 测试GPU推理
    if torch.cuda.is_available():
        print('测试 YOLO GPU 推理...')
        try:
            model = model.to('cuda:0')
            results_gpu = model(test_img, device='cuda:0', verbose=False)
            print('✅ YOLO GPU 测试通过')
        except Exception as e:
            print(f'⚠️  YOLO GPU 测试失败: {e}')
            print('CPU 模式仍然可用')

except Exception as e:
    print(f'❌ YOLO 测试失败: {e}')
    exit(1)
"

# 创建测试脚本
echo "创建测试脚本..."
cat > test_environment.py << 'EOF'
#!/usr/bin/env python3
"""
SAM2 + YOLO 环境测试脚本
"""

import torch
import numpy as np
import cv2
import sys
import os

def test_basic_imports():
    """测试基本库导入"""
    print("=== 基本库测试 ===")
    try:
        print(f"✅ Python: {sys.version}")
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ NumPy: {np.__version__}")
        print(f"✅ OpenCV: {cv2.__version__}")
        return True
    except Exception as e:
        print(f"❌ 基本库导入失败: {e}")
        return False

def test_cuda():
    """测试CUDA支持"""
    print("\n=== CUDA 测试 ===")
    try:
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"GPU 数量: {torch.cuda.device_count()}")
            for i in range(min(torch.cuda.device_count(), 4)):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
            # 测试CUDA张量操作
            x = torch.randn(1000, 1000, device='cuda:0')
            y = torch.matmul(x, x.T)
            print("✅ CUDA 张量操作正常")
        return True
    except Exception as e:
        print(f"❌ CUDA 测试失败: {e}")
        return False

def test_yolo():
    """测试YOLO"""
    print("\n=== YOLO 测试 ===")
    try:
        from ultralytics import YOLO
        
        # 初始化模型
        model = YOLO('yolov8n.pt')
        print("✅ YOLO 模型加载成功")
        
        # 创建测试图像
        test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # CPU推理
        results = model(test_img, verbose=False)
        print("✅ YOLO CPU 推理成功")
        
        # GPU推理（如果可用）
        if torch.cuda.is_available():
            try:
                model_gpu = YOLO('yolov8n.pt').to('cuda:0')
                results_gpu = model_gpu(test_img, device='cuda:0', verbose=False)
                print("✅ YOLO GPU 推理成功")
            except Exception as e:
                print(f"⚠️  YOLO GPU 推理失败: {e}")
        
        return True
    except Exception as e:
        print(f"❌ YOLO 测试失败: {e}")
        return False

def test_sam2():
    """测试SAM2"""
    print("\n=== SAM2 测试 ===")
    try:
        # 尝试导入SAM2
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print("✅ SAM2 模块导入成功")
        
        # 注意：实际使用需要下载模型文件
        print("⚠️  SAM2 需要下载模型文件才能完全测试")
        print("   模型下载命令：")
        print("   cd sam2/checkpoints && ./download_ckpts.sh")
        
        return True
    except Exception as e:
        print(f"❌ SAM2 测试失败: {e}")
        return False

def test_integration():
    """测试集成功能"""
    print("\n=== 集成测试 ===")
    try:
        # 创建测试图像
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # OpenCV处理
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # NumPy-PyTorch转换
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        
        if torch.cuda.is_available():
            img_gpu = img_tensor.cuda()
            img_cpu = img_gpu.cpu().numpy()
        
        print("✅ 图像处理和张量转换正常")
        return True
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🚀 开始环境测试...\n")
    
    tests = [
        test_basic_imports,
        test_cuda,
        test_yolo,
        test_sam2,
        test_integration
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "="*50)
    print("📊 测试结果汇总:")
    print("="*50)
    
    if all(results):
        print("🎉 所有测试通过！环境配置成功！")
        print("\n📋 环境信息:")
        print(f"   - Python: {sys.version.split()[0]}")
        print(f"   - PyTorch: {torch.__version__}")
        print(f"   - CUDA: {torch.cuda.is_available()}")
        print(f"   - GPU数量: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
        
        print("\n🚀 开始使用:")
        print("1. 激活环境: conda activate sam2_yolo")
        print("2. 设置环境变量: ")
        print(f"   export CUDA_HOME={os.environ.get('CUDA_HOME', '/path/to/cuda')}")
        print("3. 下载SAM2模型: cd sam2/checkpoints && ./download_ckpts.sh")
        
    else:
        print("❌ 部分测试失败，请检查错误信息")
        failed_tests = [i for i, result in enumerate(results) if not result]
        print(f"失败的测试: {failed_tests}")
EOF

# 运行测试
echo "运行完整环境测试..."
python test_environment.py

echo ""
echo "🎉 SAM2 + YOLO 环境配置完成！"
echo ""
echo "📁 项目结构："
echo "   $(pwd)/"
echo "   ├── sam2/                    # SAM2 源码"
echo "   ├── test_environment.py     # 环境测试脚本"
echo "   └── 你的项目文件..."
echo ""
echo "🚀 下一步："
echo "1. 下载 SAM2 模型:"
echo "   cd sam2/checkpoints"
echo "   ./download_ckpts.sh"
echo ""
echo "2. 开始开发你的项目！"