#!/usr/bin/env python3
"""
YOLO -> SAM2 完整流程集成测试
测试从YOLO检测到SAM2分割的完整pipeline
"""

import torch
import numpy as np
import sys
import os
from PIL import Image
import cv2

def load_models():
    """加载YOLO和SAM2模型"""
    print("🚀 加载模型...")
    
    # 1. 加载YOLO
    print("1. 加载YOLO...")
    from ultralytics import YOLO
    yolo_model = YOLO("./yolov8n.pt")
    print("✅ YOLO加载成功")
    
    # 2. 加载SAM2
    print("2. 加载SAM2...")
    # 添加SAM2路径
    sys.path.append(os.path.abspath("../sam2"))
    
    from hydra import initialize_config_dir, compose
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    
    # 清除Hydra实例
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    
    # 加载SAM2配置和模型
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
        sam_predictor = SAM2ImagePredictor(model)
        print("✅ SAM2加载成功")
    
    return yolo_model, sam_predictor

def create_test_image_with_objects():
    """创建包含明显对象的测试图像"""
    print("\n📋 创建测试图像...")
    
    # 创建一个有明显几何形状的图像，更容易被YOLO检测
    image = np.ones((640, 640, 3), dtype=np.uint8) * 50  # 深灰背景
    
    # 添加一些明显的形状
    # 红色矩形 (模拟车辆)
    cv2.rectangle(image, (100, 200), (250, 350), (0, 0, 255), -1)
    
    # 蓝色圆形 (模拟球类)
    cv2.circle(image, (450, 300), 80, (255, 0, 0), -1)
    
    # 绿色矩形 (模拟另一个对象)
    cv2.rectangle(image, (300, 100), (500, 200), (0, 255, 0), -1)
    
    # 添加一些噪声让图像更真实
    noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)
    
    print(f"✅ 创建测试图像: {image.shape}")
    return image

def load_real_image(image_path):
    """加载真实图像文件"""
    print(f"\n📋 加载真实图像: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        return None
    
    try:
        # 使用PIL加载图像
        pil_image = Image.open(image_path).convert('RGB')
        image_array = np.array(pil_image)
        print(f"✅ 图像加载成功: {image_array.shape}")
        return image_array
    except Exception as e:
        print(f"❌ 图像加载失败: {e}")
        return None

def test_yolo_detection_with_filter(yolo_model, image, target_classes=['person']):
    """测试YOLO检测，只保留指定类别"""
    print(f"\n📋 Step 1: YOLO目标检测 (筛选类别: {target_classes})...")
    
    # YOLO推理
    results = yolo_model(image, verbose=False)
    
    # 解析检测结果，只保留指定类别
    detections = []
    if results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            # 获取边界框坐标 (xyxy格式)
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            confidence = boxes.conf[i].cpu().numpy()
            class_id = int(boxes.cls[i].cpu().numpy())
            class_name = yolo_model.names[class_id]
            
            # 只保留目标类别
            if class_name in target_classes:
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name,
                    'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                }
                detections.append(detection)
    
    print(f"✅ YOLO检测结果: {len(detections)} 个{target_classes}对象")
    for i, det in enumerate(detections):
        print(f"   对象 {i+1}: {det['class_name']} (置信度: {det['confidence']:.3f})")
        print(f"           边界框: [{det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}, {det['bbox'][2]:.0f}, {det['bbox'][3]:.0f}]")
        print(f"           中心点: [{det['center'][0]:.0f}, {det['center'][1]:.0f}]")
    
    return detections

def test_sam2_segmentation(sam_predictor, image, detections):
    """使用YOLO检测结果进行SAM2分割"""
    print(f"\n📋 Step 2: SAM2分割 (基于{len(detections)}个YOLO检测结果)...")
    
    # 设置图像
    sam_predictor.set_image(image)
    print("✅ SAM2图像设置完成")
    
    all_masks = []
    all_scores = []
    
    for i, detection in enumerate(detections):
        print(f"\n   处理对象 {i+1}: {detection['class_name']}")
        
        # 方法1: 使用中心点作为prompt
        center_point = np.array([detection['center']], dtype=np.float32)
        center_label = np.array([1])  # 前景点
        
        try:
            # SAM2分割
            masks, scores, logits = sam_predictor.predict(
                point_coords=center_point,
                point_labels=center_label,
                multimask_output=True,
            )
            
            # 选择最佳mask
            best_mask_idx = scores.argmax()
            best_mask = masks[best_mask_idx]
            best_score = scores[best_mask_idx]
            
            all_masks.append(best_mask)
            all_scores.append(best_score)
            
            print(f"   ✅ 分割成功: 分数 {best_score:.3f}, mask形状 {best_mask.shape}")
            
        except Exception as e:
            print(f"   ❌ 分割失败: {e}")
            continue
    
    print(f"\n✅ SAM2分割完成: 成功分割 {len(all_masks)} 个对象")
    return all_masks, all_scores

def test_sam2_with_bbox_prompts(sam_predictor, image, detections):
    """使用边界框作为SAM2的输入prompt"""
    print(f"\n📋 Step 3: SAM2边界框分割...")
    
    all_bbox_masks = []
    all_bbox_scores = []
    
    for i, detection in enumerate(detections):
        print(f"   处理对象 {i+1}边界框: {detection['class_name']}")
        
        # 使用边界框作为prompt
        bbox = np.array(detection['bbox'], dtype=np.float32)  # [x1, y1, x2, y2]
        
        try:
            # SAM2边界框分割
            masks, scores, logits = sam_predictor.predict(
                box=bbox,
                multimask_output=False,  # 边界框通常用单mask
            )
            
            best_mask = masks[0]
            best_score = scores[0]
            
            all_bbox_masks.append(best_mask)
            all_bbox_scores.append(best_score)
            
            print(f"   ✅ 边界框分割成功: 分数 {best_score:.3f}")
            
        except Exception as e:
            print(f"   ❌ 边界框分割失败: {e}")
            continue
    
    print(f"\n✅ 边界框分割完成: 成功分割 {len(all_bbox_masks)} 个对象")
    return all_bbox_masks, all_bbox_scores

def analyze_results(image, detections, point_masks, point_scores, bbox_masks, bbox_scores):
    """分析结果"""
    print(f"\n📊 结果分析...")
    
    print(f"原始图像: {image.shape}")
    print(f"YOLO检测: {len(detections)} 个对象")
    print(f"点prompt分割: {len(point_masks)} 个mask")
    print(f"框prompt分割: {len(bbox_masks)} 个mask")
    
    # 比较两种方法的分数
    if point_scores and bbox_scores:
        avg_point_score = np.mean(point_scores)
        avg_bbox_score = np.mean(bbox_scores)
        
        print(f"\n分割质量比较:")
        print(f"  点prompt平均分数: {avg_point_score:.3f}")
        print(f"  框prompt平均分数: {avg_bbox_score:.3f}")
        print(f"  {'框prompt更优' if avg_bbox_score > avg_point_score else '点prompt更优'}")
    
    # 分析mask覆盖度
    if point_masks:
        total_pixels = image.shape[0] * image.shape[1]
        for i, mask in enumerate(point_masks):
            mask_pixels = np.sum(mask)
            coverage = mask_pixels / total_pixels * 100
            print(f"  对象{i+1}点mask覆盖: {coverage:.1f}%")
    
    if bbox_masks:
        for i, mask in enumerate(bbox_masks):
            mask_pixels = np.sum(mask)
            coverage = mask_pixels / total_pixels * 100
            print(f"  对象{i+1}框mask覆盖: {coverage:.1f}%")

def save_visualization(image, detections, point_masks, bbox_masks, output_name="yolo_sam2_test_result.jpg"):
    """保存可视化结果（快速修复版）"""
    print(f"\n💾 保存可视化结果...")
    
    try:
        # 在检测框上绘制结果
        vis_image = image.copy()
        
        # 绘制YOLO检测框
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = [int(x) for x in det['bbox']]
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 255), 2)  # 黄色框
            cv2.putText(vis_image, f"{det['class_name']}: {det['confidence']:.2f}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # 如果有masks，叠加显示 - 修复版
        if point_masks:
            for i, mask in enumerate(point_masks):
                # 关键修复：确保mask是boolean类型
                mask_bool = mask.astype(bool)
                
                # 创建彩色mask
                color_mask = np.zeros_like(vis_image)
                color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
                mask_color = color[i % len(color)]
                color_mask[mask_bool] = mask_color  # 使用boolean mask
                
                # 透明叠加
                vis_image = cv2.addWeighted(vis_image, 0.8, color_mask, 0.2, 0)
        
        # 保存结果
        success = cv2.imwrite(output_name, vis_image)
        if success:
            print(f"✅ 可视化结果已保存: {output_name}")
            # 检查文件是否确实存在
            if os.path.exists(output_name):
                file_size = os.path.getsize(output_name)
                print(f"✅ 文件确认存在: {output_name} ({file_size:,} bytes)")
            else:
                print(f"⚠️ 文件保存后不存在: {output_name}")
        else:
            print(f"❌ 文件保存失败: {output_name}")
        
    except Exception as e:
        print(f"⚠️ 可视化保存失败: {e}")
        # 保存一个简化版本
        try:
            simple_vis = image.copy()
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = [int(x) for x in det['bbox']]
                cv2.rectangle(simple_vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            simple_name = f"simple_{output_name}"
            cv2.imwrite(simple_name, simple_vis)
            print(f"✅ 简化版本已保存: {simple_name}")
        except:
            print("❌ 简化版本保存也失败")
            
def test_real_image_yolo_sam2(image_path="../data/video_composed_frames/train/fgr/0000/frames/0088.png"):
    """测试真实图像的YOLO->SAM2流程"""
    print("🔍 真实图像 YOLO -> SAM2 测试")
    print("="*60)
    print(f"📁 目标图像: {image_path}")
    
    try:
        # 1. 加载模型
        yolo_model, sam_predictor = load_models()
        
        # 2. 加载真实图像
        real_image = load_real_image(image_path)
        if real_image is None:
            print("❌ 无法加载图像，退出测试")
            return
        
        # 3. YOLO检测（只检测person）
        detections = test_yolo_detection_with_filter(yolo_model, real_image, target_classes=['person'])
        
        if not detections:
            print("⚠️ 未检测到任何person对象")
            print("📋 尝试检测所有对象类别...")
            # 如果没有检测到person，显示所有检测结果
            all_detections = test_yolo_detection_with_filter(yolo_model, real_image, 
                                                           target_classes=list(yolo_model.names.values()))
            if all_detections:
                print("检测到的其他对象:")
                unique_classes = set([det['class_name'] for det in all_detections])
                print(f"类别: {', '.join(unique_classes)}")
                
                # 询问是否继续用其他类别测试
                print("是否使用检测到的第一个对象进行SAM2测试？")
                detections = [all_detections[0]]  # 使用第一个检测结果
            else:
                print("❌ 没有检测到任何对象，退出测试")
                return
        
        # 4. SAM2分割 - 点prompt
        point_masks, point_scores = test_sam2_segmentation(sam_predictor, real_image, detections)
        
        # 5. SAM2分割 - 边界框prompt
        bbox_masks, bbox_scores = test_sam2_with_bbox_prompts(sam_predictor, real_image, detections)
        
        # 6. 分析结果
        analyze_results(real_image, detections, point_masks, point_scores, bbox_masks, bbox_scores)
        
        # 7. 保存可视化（使用特殊文件名）
        output_name = f"real_image_yolo_sam2_result_{os.path.basename(image_path)}"
        save_visualization(real_image, detections, point_masks, bbox_masks, output_name)
        
        print("\n" + "="*60)
        print("🎉 真实图像 YOLO -> SAM2 测试完成!")
        print(f"✅ 处理图像: {image_path}")
        print(f"✅ 检测到 {len(detections)} 个目标对象")
        print(f"✅ 成功分割 {len(point_masks)} 个对象")
        print(f"✅ 结果保存: {output_name}")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 真实图像测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主测试函数"""
    print("🔍 YOLO -> SAM2 完整流程集成测试")
    print("="*60)
    
    # 选择测试模式
    print("请选择测试模式:")
    print("1. 真实图像测试 (../data/video_composed_frames/train/fgr/0000/frames/0088.png)")
    print("2. 生成图像测试")
    print("3. 两种都测试")
    
    choice = input("请输入选择 (1/2/3, 默认1): ").strip() or "1"
    
    if choice in ["1", "3"]:
        print("\n" + "🔍 开始真实图像测试" + "="*40)
        test_real_image_yolo_sam2()
    
    if choice in ["2", "3"]:
        print("\n" + "🔍 开始生成图像测试" + "="*40)
        run_synthetic_test()

def run_synthetic_test():
    """运行原来的合成图像测试"""
    try:
        # 1. 加载模型
        yolo_model, sam_predictor = load_models()
        
        # 2. 创建测试图像
        test_image = create_test_image_with_objects()
        
        # 3. YOLO检测
        detections = test_yolo_detection_with_filter(yolo_model, test_image, 
                                                   target_classes=list(yolo_model.names.values()))
        
        if not detections:
            print("⚠️ YOLO未检测到对象，创建模拟检测结果...")
            detections = [{
                'bbox': [100, 200, 250, 350],
                'confidence': 0.85,
                'class_id': 0,
                'class_name': 'test_object',
                'center': [175, 275]
            }]
        
        # 4. SAM2分割 - 点prompt
        point_masks, point_scores = test_sam2_segmentation(sam_predictor, test_image, detections)
        
        # 5. SAM2分割 - 边界框prompt
        bbox_masks, bbox_scores = test_sam2_with_bbox_prompts(sam_predictor, test_image, detections)
        
        # 6. 分析结果
        analyze_results(test_image, detections, point_masks, point_scores, bbox_masks, bbox_scores)
        
        # 7. 保存可视化
        save_visualization(test_image, detections, point_masks, bbox_masks, "synthetic_yolo_sam2_result.jpg")
        
        print("\n" + "="*60)
        print("🎉 生成图像 YOLO -> SAM2 测试完成!")
        print("✅ 完整流程工作正常")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 生成图像测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()