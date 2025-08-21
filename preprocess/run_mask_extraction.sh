#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Script information
SCRIPT_NAME="Enhanced YOLO-SAM2 Mask Extraction"
VERSION="2.1 (CUDA Fixed)"

# Print banner
echo "=========================================="
echo "  $SCRIPT_NAME v$VERSION"
echo "=========================================="

# Default parameters
MODE="first_frame"
DEBUG_MODE=false
WORKERS=""
RESUME=false
FORCE_CLEAN=false

# Function to show usage
show_usage() {
    echo "Usage: $0 [MODE] [OPTIONS]"
    echo ""
    echo "Modes:"
    echo "  first_frame    Extract mask for first frame only (default)"
    echo "  all_frames     Extract masks for all frames"
    echo ""
    echo "Options:"
    echo "  --debug        Enable debug output"
    echo "  --workers N    Set number of workers (default: auto)"
    echo "  --resume       Resume from previous incomplete run"
    echo "  --clean        Clean up orphan processes before starting"
    echo "  --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 first_frame"
    echo "  $0 all_frames --debug --workers 4"
    echo "  $0 first_frame --resume"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        first_frame|all_frames)
            MODE="$1"
            shift
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --clean)
            FORCE_CLEAN=true
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option '$1'"
            show_usage
            exit 1
            ;;
    esac
done

# Validate workers parameter
if [[ -n "$WORKERS" ]] && ! [[ "$WORKERS" =~ ^[0-9]+$ ]]; then
    echo "Error: Workers must be a positive integer"
    exit 1
fi

# Environment setup
echo "[INFO] Setting up environment..."

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "[INFO] Conda detected, activating environment..."
    source ~/.bashrc
    # Uncomment the next line if you have a specific environment
    # conda activate sam2_matanyone
else
    echo "[WARNING] Conda not found, using system Python"
fi

# Check Python and required packages
echo "[INFO] Checking Python environment..."
python3 -c "
import sys
print(f'Python version: {sys.version}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA devices: {torch.cuda.device_count()}')
        print(f'Current device: {torch.cuda.current_device()}')
        print(f'CUDA version: {torch.version.cuda}')
except ImportError as e:
    print(f'WARNING: PyTorch not found: {e}')
    sys.exit(1)

try:
    import ultralytics
    print(f'Ultralytics version: {ultralytics.__version__}')
except ImportError as e:
    print(f'WARNING: Ultralytics not found: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "Error: Python environment check failed"
    exit 1
fi

echo "[INFO] Starting mask extraction in ${MODE} mode at $(date)"

# 修复的内存优化配置
# 移除可能不兼容的expandable_segments配置
echo "[INFO] Setting up CUDA memory configuration..."

# 检查PyTorch版本并设置合适的内存配置
PYTORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
echo "[INFO] Detected PyTorch version: $PYTORCH_VERSION"

# 基于PyTorch版本设置内存配置
if [[ "$PYTORCH_VERSION" > "2.0" ]]; then
    echo "[INFO] Using modern PyTorch memory configuration"
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
else
    echo "[INFO] Using legacy PyTorch memory configuration"
    unset PYTORCH_CUDA_ALLOC_CONF
fi

# 其他CUDA优化设置
export CUDA_LAUNCH_BLOCKING=0  # For better performance
export TORCH_CUDNN_V8_API_ENABLED=1  # Enable cuDNN v8 API if available

# Paths
SCRIPT_PATH="yolo_sam2_mask_extract.py"
DATA_DIR="../data/video_composed_frames"

# Validate paths
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script not found at $SCRIPT_PATH"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found at $DATA_DIR"
    exit 1
fi

# Create log directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Build command arguments
PYTHON_ARGS="--data_dir \"$DATA_DIR\" --mode \"$MODE\""

if [ "$DEBUG_MODE" = true ]; then
    PYTHON_ARGS="$PYTHON_ARGS --debug"
    echo "[INFO] Debug mode enabled"
fi

if [[ -n "$WORKERS" ]]; then
    PYTHON_ARGS="$PYTHON_ARGS --workers $WORKERS"
    echo "[INFO] Using $WORKERS workers"
fi

# Clean up function
cleanup_processes() {
    echo "[INFO] Cleaning up processes..."
    pkill -f "yolo_sam2_mask_extract.py" 2>/dev/null || true
    sleep 2
}

# Handle interruption
trap 'echo "[INFO] Script interrupted. Cleaning up..."; cleanup_processes; exit 130' INT TERM

# Force cleanup if requested
if [ "$FORCE_CLEAN" = true ]; then
    echo "[INFO] Force cleaning orphan processes..."
    cleanup_processes
fi

# Create comprehensive log file
LOG_FILE="${LOG_DIR}/mask_extraction_${MODE}_${TIMESTAMP}.log"
STATUS_FILE="${LOG_DIR}/status_${MODE}_${TIMESTAMP}.txt"

# Function to log system stats
log_system_stats() {
    echo "=== System Information ===" >> "$STATUS_FILE"
    echo "Timestamp: $(date)" >> "$STATUS_FILE"
    echo "Mode: $MODE" >> "$STATUS_FILE"
    echo "Debug: $DEBUG_MODE" >> "$STATUS_FILE"
    echo "Workers: ${WORKERS:-auto}" >> "$STATUS_FILE"
    echo "" >> "$STATUS_FILE"
    
    echo "=== GPU Information ===" >> "$STATUS_FILE"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi >> "$STATUS_FILE" 2>&1
    else
        echo "nvidia-smi not available" >> "$STATUS_FILE"
    fi
    echo "" >> "$STATUS_FILE"
    
    echo "=== Memory Information ===" >> "$STATUS_FILE"
    free -h >> "$STATUS_FILE" 2>&1
    echo "" >> "$STATUS_FILE"
    
    echo "=== Disk Space ===" >> "$STATUS_FILE"
    df -h "$DATA_DIR" >> "$STATUS_FILE" 2>&1
    echo "" >> "$STATUS_FILE"
}

# Log initial system stats
log_system_stats

# Check for resume capability
if [ "$RESUME" = true ]; then
    echo "[INFO] Resume mode enabled - checking for previous progress..."
    for split in train test; do
        if [ -f "completed_masks_all_frames_${split}.json" ]; then
            completed_count=$(python3 -c "
import json
try:
    with open('completed_masks_all_frames_${split}.json') as f:
        data = json.load(f)
        print(len(data.get('completed_videos', [])))
except:
    print(0)
")
            echo "[INFO] Found $completed_count completed videos in $split split"
        fi
    done
fi

# Pre-flight checks
echo "[INFO] Performing pre-flight checks..."

# Check available disk space (at least 10GB recommended)
available_space=$(df "$DATA_DIR" | tail -1 | awk '{print $4}')
if [ "$available_space" -lt 10485760 ]; then  # 10GB in KB
    echo "[WARNING] Low disk space detected (less than 10GB available)"
    echo "[WARNING] Consider freeing up space before proceeding"
fi

# Show configuration summary
echo ""
echo "=== Configuration Summary ==="
echo "Mode: $MODE"
echo "Data directory: $DATA_DIR"
echo "Debug mode: $DEBUG_MODE"
echo "Workers: ${WORKERS:-auto-detected}"
echo "Resume: $RESUME"
echo "Log file: $LOG_FILE"
echo "Status file: $STATUS_FILE"
echo "============================"
echo ""

# Confirmation prompt for all_frames mode
if [ "$MODE" = "all_frames" ] && [ "$RESUME" = false ]; then
    echo "[WARNING] You are about to process ALL frames in the dataset."
    echo "[WARNING] This may take a very long time and use significant resources."
    read -p "Are you sure you want to continue? (y/N): " -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "[INFO] Operation cancelled by user"
        exit 0
    fi
fi

# Start processing
echo "[INFO] Running ${MODE} mask extraction..."
start_time=$(date +%s)

# Run the script with proper error handling
set +e  # Don't exit on error for this command
eval "python3 \"$SCRIPT_PATH\" $PYTHON_ARGS" 2>&1 | tee "$LOG_FILE"
exit_code=$?
set -e

end_time=$(date +%s)
duration=$((end_time - start_time))

# Log final stats
echo "" >> "$STATUS_FILE"
echo "=== Execution Summary ===" >> "$STATUS_FILE"
echo "Start time: $(date -d @$start_time)" >> "$STATUS_FILE"
echo "End time: $(date -d @$end_time)" >> "$STATUS_FILE"
echo "Duration: $((duration / 3600))h $((duration % 3600 / 60))m $((duration % 60))s" >> "$STATUS_FILE"
echo "Exit code: $exit_code" >> "$STATUS_FILE"

# Final system stats
echo "" >> "$STATUS_FILE"
echo "=== Final System Stats ===" >> "$STATUS_FILE"
log_system_stats

# Print results summary
echo ""
echo "=== Execution Summary ==="
echo "Mode: $MODE"
echo "Duration: $((duration / 3600))h $((duration % 3600 / 60))m $((duration % 60))s"
echo "Exit code: $exit_code"
echo "Log file: $LOG_FILE"
echo "Status file: $STATUS_FILE"

# Analyze results
if [ $exit_code -eq 0 ]; then
    echo "Status: SUCCESS"
    
    # Try to extract completion statistics
    echo ""
    echo "=== Completion Statistics ==="
    for split in train test; do
        if [ -f "completed_masks_all_frames_${split}.json" ]; then
            completed_count=$(python3 -c "
import json
try:
    with open('completed_masks_all_frames_${split}.json') as f:
        data = json.load(f)
        completed = len(data.get('completed_videos', []))
        print(f'$split: {completed} videos completed')
        
        # Calculate average processing time if available
        stats = data.get('stats', {})
        if stats:
            times = [s.get('processing_time', 0) for s in stats.values() if s.get('processing_time')]
            if times:
                avg_time = sum(times) / len(times)
                print(f'$split: Average processing time: {avg_time:.1f}s per video')
except Exception as e:
    print(f'$split: Error reading stats - {e}')
" 2>/dev/null)
            echo "$completed_count"
        fi
    done
    
elif [ $exit_code -eq 130 ]; then
    echo "Status: INTERRUPTED"
    echo "The process was interrupted by user"
else
    echo "Status: FAILED"
    echo "Check the log file for error details: $LOG_FILE"
fi

echo "=========================="
echo "[INFO] Finished mask extraction task at $(date)"

exit $exit_code