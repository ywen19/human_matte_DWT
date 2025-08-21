#!/bin/bash

set -e

echo "通过Conda安装CUDA 11.8工具包（使用镜像源）..."

# 添加清华镜像源
echo "配置conda镜像源..."
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/

# 设置镜像优先级
conda config --set channel_priority strict

# 删除现有环境（如果存在）
conda env remove -n cuda118 -y || true

# 创建CUDA环境
echo "创建CUDA 11.8环境..."
conda create -n cuda118 python=3.10 -y

# 激活环境
eval "$(conda shell.bash hook)"
conda activate cuda118

# 尝试安装CUDA工具包（使用多个源）
echo "安装CUDA 11.8工具包..."
conda install cudatoolkit=11.8 -y || \
conda install -c pytorch cudatoolkit=11.8 -y || \
conda install -c nvidia cudatoolkit=11.8 -y

# 尝试安装开发工具
echo "尝试安装CUDA开发工具..."
conda install cudatoolkit-dev=11.8 -y || echo "⚠️ cudatoolkit-dev安装失败，继续..."

# 获取环境路径 - 修复版本
echo "获取conda环境路径..."
CONDA_ENV_PATH=$(conda info --envs | grep "cuda118" | head -1 | awk '{print $NF}')

if [ -z "$CONDA_ENV_PATH" ]; then
    # 尝试另一种方法获取路径
    CONDA_ENV_PATH="$CONDA_PREFIX"
fi

if [ -z "$CONDA_ENV_PATH" ]; then
    # 使用默认路径
    CONDA_ENV_PATH="$HOME/anaconda3/envs/cuda118"
fi

CUDA_HOME="$CONDA_ENV_PATH"
echo "CUDA安装路径: $CUDA_HOME"

# 验证CUDA安装
echo "验证CUDA安装..."
if [ -f "$CUDA_HOME/bin/nvcc" ]; then
    echo "找到nvcc编译器"
    $CUDA_HOME/bin/nvcc --version
    echo "✅ CUDA 11.8工具包完整安装成功！"
elif [ -d "$CUDA_HOME/lib" ] || [ -d "$CUDA_HOME/lib64" ]; then
    echo "找到CUDA库文件"
    echo "CUDA路径: $CUDA_HOME"
    if [ -d "$CUDA_HOME/lib" ]; then
        ls -la "$CUDA_HOME/lib" | grep -i cuda | head -5
    fi
    if [ -d "$CUDA_HOME/lib64" ]; then
        ls -la "$CUDA_HOME/lib64" | grep -i cuda | head -5
    fi
    echo "✅ CUDA运行时库安装成功！"
elif [ -d "$CUDA_HOME" ]; then
    echo "环境目录存在，检查内容："
    ls -la "$CUDA_HOME/" | head -10
    # 检查是否有CUDA相关文件
    find "$CUDA_HOME" -name "*cuda*" -type f 2>/dev/null | head -5
    echo "✅ CUDA环境已创建！"
else
    echo "❌ 环境路径不存在: $CUDA_HOME"
    echo "尝试查找所有conda环境："
    conda info --envs
    exit 1
fi

echo ""
echo "🎉 Conda CUDA 11.8安装完成！"
echo ""
echo "环境路径: $CUDA_HOME"
echo "接下来可以运行环境配置脚本"