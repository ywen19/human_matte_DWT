#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import torch
import numpy as np

def check_cuda_environment():
    """检查CUDA环境"""
    print("=" * 60)
    print("CUDA环境诊断")
    print("=" * 60)
    
    # 1. 系统CUDA版本
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("系统CUDA版本:")
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    print(f"  {line.strip()}")
        else:
            print("❌ 无法获取系统CUDA版本")
    except FileNotFoundError:
        print("❌ nvcc未找到，CUDA可能未正确安装")
    
    # 2. PyTorch CUDA版本
    print(f"\nPyTorch信息:")
    print(f"  PyTorch版本: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  PyTorch CUDA版本: {torch.version.cuda}")
        print(f"  cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"  GPU设备数量: {torch.cuda.device_count()}")
        
        # 显示所有GPU信息
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    内存: {props.total_memory / 1024**3:.1f} GB")
            print(f"    计算能力: {props.major}.{props.minor}")
    
    # 3. NumPy信息
    print(f"\nNumPy信息:")
    print(f"  NumPy版本: {np.__version__}")
    
    # 4. 环境变量
    print(f"\n环境变量:")
    cuda_home = os.environ.get('CUDA_HOME', '未设置')
    print(f"  CUDA_HOME: {cuda_home}")
    ld_path = os.environ.get('LD_LIBRARY_PATH', '未设置')
    print(f"  LD_LIBRARY_PATH: {'已设置' if ld_path != '未设置' else '未设置'}")

def check_package_versions():
    """检查关键包版本"""
    print("\n" + "=" * 60)
    print("关键包版本检查")
    print("=" * 60)
    
    # 更新包列表，匹配您的环境
    packages = [
        'torch', 'torchvision', 'torchaudio', 'numpy', 'ultralytics', 
        'opencv-python', 'pillow', 'matplotlib', 'hydra-core', 'omegaconf'
    ]
    
    for package in packages:
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'show', package], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                name = next((line.split(': ')[1] for line in lines if line.startswith('Name:')), 'Unknown')
                version = next((line.split(': ')[1] for line in lines if line.startswith('Version:')), 'Unknown')
                print(f"  {name}: {version}")
            else:
                print(f"  {package}: 未安装")
        except Exception as e:
            print(f"  {package}: 检查失败 - {e}")

def test_cuda_compatibility():
    """测试CUDA兼容性"""
    print("\n" + "=" * 60)
    print("CUDA兼容性测试")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    try:
        # 1. 基本CUDA操作
        print("1. 测试基本CUDA操作...")
        a = torch.randn(1000, 1000).cuda()
        b = torch.randn(1000, 1000).cuda()
        c = torch.mm(a, b)
        print("  ✅ 基本CUDA操作成功")
        
        # 2. 多GPU测试（如果有多个GPU）
        if torch.cuda.device_count() > 1:
            print("2. 测试多GPU操作...")
            device_0 = torch.device('cuda:0')
            device_1 = torch.device('cuda:1')
            x = torch.randn(100, 100, device=device_0)
            y = torch.randn(100, 100, device=device_1)
            print(f"  ✅ 多GPU操作成功 (共{torch.cuda.device_count()}个GPU)")
        
        # 3. CPU-GPU数据传输
        print("3. 测试CPU-GPU数据传输...")
        cpu_tensor = torch.randn(1000, 1000)
        gpu_tensor = cpu_tensor.cuda()
        back_to_cpu = gpu_tensor.cpu()
        print("  ✅ CPU-GPU数据传输成功")
        
        # 4. NumPy-PyTorch互转（您的环境中的关键测试）
        print("4. 测试NumPy-PyTorch互转...")
        np_array = np.random.randn(100, 100).astype(np.float32)
        torch_tensor = torch.from_numpy(np_array)
        back_to_numpy = torch_tensor.numpy()
        
        # 测试CUDA张量转换
        cuda_tensor = torch_tensor.cuda()
        cpu_numpy = cuda_tensor.cpu().numpy()
        print("  ✅ NumPy-PyTorch互转成功")
        
        # 5. CUDA内存管理
        print("5. 测试CUDA内存管理...")
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            reserved = torch.cuda.memory_reserved(i) / 1024**2
            print(f"  GPU {i} 内存: 已分配 {allocated:.1f} MB, 已保留 {reserved:.1f} MB")
        print("  ✅ CUDA内存管理正常")
        
        return True
        
    except Exception as e:
        print(f"❌ CUDA兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sam2_yolo_environment():
    """测试SAM2+YOLO环境（从preprocess目录运行）"""
    print("\n" + "=" * 60)
    print("SAM2 + YOLO 环境测试 (从preprocess目录)")
    print("=" * 60)
    
    # 1. 测试YOLO
    print("1. 测试YOLO...")
    try:
        # 在preprocess目录中，YOLO模型就在当前目录
        yolo_path = "./yolov8n.pt"  # 当前目录下
        if os.path.exists(yolo_path):
            from ultralytics import YOLO
            model = YOLO(yolo_path)
            print(f"  ✅ YOLO模型加载成功: {yolo_path}")
            
            # 快速推理测试
            test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            results = model(test_img, verbose=False)
            print(f"  ✅ YOLO推理成功")
        else:
            print(f"  ❌ YOLO模型文件不存在: {yolo_path}")
    except Exception as e:
        print(f"  ❌ YOLO测试失败: {e}")
    
    # 2. 测试SAM2环境
    print("2. 测试SAM2环境...")
    try:
        # 从preprocess目录，SAM2路径是 ../sam2
        sam2_path = "../sam2"
        config_path = "../sam2/sam2/configs/sam2_hiera_l.yaml"
        model_path = "../local_sam2_hiera_large/sam2_hiera_large.pt"
        
        if os.path.exists(sam2_path):
            print(f"  ✅ SAM2代码目录存在: {sam2_path}")
        else:
            print(f"  ❌ SAM2代码目录不存在: {sam2_path}")
            
        if os.path.exists(config_path):
            print(f"  ✅ SAM2配置文件存在: {config_path}")
        else:
            print(f"  ❌ SAM2配置文件不存在: {config_path}")
            
        if os.path.exists(model_path):
            size = os.path.getsize(model_path) / 1024**2
            print(f"  ✅ SAM2模型文件存在: {model_path} ({size:.1f} MB)")
        else:
            print(f"  ❌ SAM2模型文件不存在: {model_path}")
            
        # 尝试导入SAM2（如果路径正确）
        if os.path.exists(sam2_path):
            sys.path.append(os.path.abspath(sam2_path))
            try:
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                print(f"  ✅ SAM2模块导入成功")
            except Exception as e:
                print(f"  ⚠️ SAM2模块导入失败: {e}")
                
    except Exception as e:
        print(f"  ❌ SAM2环境检查失败: {e}")
    
    # 3. 显示当前目录信息
    print("3. 当前目录结构检查...")
    current_dir = os.getcwd()
    print(f"  当前工作目录: {current_dir}")
    
    # 列出当前目录的文件
    files = [f for f in os.listdir('.') if f.endswith(('.pt', '.py'))]
    if files:
        print(f"  当前目录文件:")
        for f in files:
            print(f"    - {f}")
    
    # 检查父目录
    parent_files = []
    try:
        parent_files = [f for f in os.listdir('..') if os.path.isdir(os.path.join('..', f))]
        print(f"  父目录包含: {', '.join(parent_files[:5])}{'...' if len(parent_files) > 5 else ''}")
    except:
        pass

def suggest_fixes():
    """建议修复方案"""
    print("\n" + "=" * 60)
    print("优化建议")
    print("=" * 60)
    
    # 检查PyTorch和CUDA版本匹配
    pytorch_version = torch.__version__
    cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"
    
    print(f"当前配置: PyTorch {pytorch_version} + CUDA {cuda_version}")
    
    # 根据您的成功配置给出建议
    if "+cu118" in pytorch_version and "1.26" in np.__version__:
        print("\n🎉 当前配置看起来很好！")
        print("✅ PyTorch 2.0.1+cu118 - 正确")
        print("✅ NumPy 1.26.x - 兼容")
        print("✅ CUDA 11.8 支持")
        
        print("\n💡 性能优化建议:")
        print("1. 确保设置了正确的环境变量:")
        print("   export CUDA_HOME=/home/wy/anaconda3/envs/cuda118")
        print("   export PATH=$CUDA_HOME/bin:$PATH")
        print("   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH")
        
        print("\n2. 如果遇到任何问题，重启环境:")
        print("   conda deactivate && conda activate sam2_yolo")
        
    else:
        print("\n🔧 修复方案 (按优先级排序):")
        
        print("\n1. 恢复到成功配置:")
        print("   pip uninstall torch torchvision torchaudio")
        print("   pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \\")
        print("       --index-url https://download.pytorch.org/whl/cu118")
        
        print("\n2. 恢复NumPy版本:")
        print("   pip install numpy==1.26.4 --force-reinstall")

def create_test_script():
    """创建针对preprocess目录的测试脚本"""
    print("\n" + "=" * 60)
    print("创建环境测试脚本 (preprocess目录版本)")
    print("=" * 60)
    
    test_script = '''#!/usr/bin/env python3
"""
SAM2 + YOLO 环境快速测试脚本
专为preprocess目录运行设计
"""

import torch
import numpy as np
import sys
import os

def quick_test():
    """快速环境测试 (从preprocess目录)"""
    print("🚀 开始快速环境测试... (从preprocess目录)")
    print(f"📁 当前工作目录: {os.getcwd()}")
    
    # 1. 基础环境检查
    print(f"\\n✅ Python: {sys.version.split()[0]}")
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ NumPy: {np.__version__}")
    print(f"✅ CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU数量: {torch.cuda.device_count()}")
    
    # 2. YOLO测试 (当前目录)
    print("\\n📋 测试YOLO...")
    try:
        from ultralytics import YOLO
        # 在preprocess目录中，模型就在当前目录
        model = YOLO("./yolov8n.pt")
        
        # 创建测试图像
        test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(test_img, verbose=False)
        print("✅ YOLO测试通过")
    except Exception as e:
        print(f"❌ YOLO测试失败: {e}")
    
    # 3. SAM2测试 (父目录)
    print("\\n📋 测试SAM2...")
    try:
        # 从preprocess目录，SAM2在父目录
        sys.path.append(os.path.abspath("../sam2"))
        
        from hydra import initialize_config_dir, compose
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        # 清除Hydra实例
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
        
        # 加载SAM2
        config_dir = os.path.abspath("../sam2/sam2/configs")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="sam2_hiera_l.yaml")
            OmegaConf.resolve(cfg)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = instantiate(cfg.model, _recursive_=True)
            
            # 加载权重
            model_path = "../local_sam2_hiera_large/sam2_hiera_large.pt"
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model"])
            
            model = model.to(device).eval()
            predictor = SAM2ImagePredictor(model)
            print("✅ SAM2测试通过")
            
    except Exception as e:
        print(f"❌ SAM2测试失败: {e}")
    
    # 4. 集成测试
    print("\\n📋 集成测试...")
    try:
        # 使用相同的测试图像
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # YOLO检测
        yolo_model = YOLO("./yolov8n.pt")
        yolo_results = yolo_model(test_image, verbose=False)
        detections = yolo_results[0].boxes
        num_detections = len(detections) if detections is not None else 0
        
        # SAM2分割
        predictor.set_image(test_image)
        input_point = np.array([[320, 240]])
        input_label = np.array([1])
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        
        print(f"✅ 集成测试成功!")
        print(f"   YOLO检测: {num_detections} 个对象")
        print(f"   SAM2分割: {len(masks)} 个mask，最佳分数: {scores.max():.3f}")
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
    
    print("\\n🎉 环境测试完成!")

if __name__ == "__main__":
    quick_test()
'''
    
    with open('quick_env_test_preprocess.py', 'w') as f:
        f.write(test_script)
    
    print("✅ 已创建 quick_env_test_preprocess.py")
    print("适合在preprocess目录运行: python quick_env_test_preprocess.py")

def main():
    print("🔍 SAM2 + YOLO 环境诊断开始...\n")
    
    # 1. 检查CUDA环境
    check_cuda_environment()
    
    # 2. 检查包版本
    check_package_versions()
    
    # 3. 测试CUDA兼容性
    cuda_ok = test_cuda_compatibility()
    
    # 4. 测试SAM2+YOLO环境（新增）
    test_sam2_yolo_environment()
    
    # 5. 提供修复建议
    suggest_fixes()
    
    # 6. 创建测试脚本
    create_test_script()
    
    print(f"\n{'='*60}")
    print("诊断完成!")
    if cuda_ok:
        print("✅ CUDA环境正常")
    else:
        print("⚠️  检测到CUDA兼容性问题")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()