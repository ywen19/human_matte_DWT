#!/usr/bin/env python3
"""
test if env for yolo+sam2 is configured successfully
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
    print(f"\n✅ Python: {sys.version.split()[0]}")
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ NumPy: {np.__version__}")
    print(f"✅ CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU数量: {torch.cuda.device_count()}")
    
    # 2. YOLO测试 (当前目录)
    print("\n📋 测试YOLO...")
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
    print("\n📋 测试SAM2...")
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
    print("\n📋 集成测试...")
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
    
    print("\n🎉 环境测试完成!")

if __name__ == "__main__":
    quick_test()
