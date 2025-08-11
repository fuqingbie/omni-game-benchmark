#!/bin/bash

# Maze navigation test startup script
# Usage: ./run_test.sh [model_type] [options]

set -e  # Exit on error

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default parameters
MODEL_TYPE="openai"
DIFFICULTY="easy"
ROUNDS=5
MAX_STEPS=500
SPEED=1.0
RESULTS_DIR="results"
INTERACTIVE=false
HEADLESS=false
SEED=""
SEQUENTIAL_SEEDS=true

# Help information
show_help() {
    echo "Maze navigation test startup script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -m, --model MODEL_TYPE    Specify model type (openai|baichuan) [default: openai]"
    echo "  -d, --difficulty LEVEL    Specify difficulty (easy|medium|hard) [default: easy]"
    echo "  -r, --rounds NUM          Specify number of test rounds [default: 5]"
    echo "  -s, --max-steps NUM       Specify maximum steps per round [default: 500]"
    echo "  --speed SPEED             Specify auto-run speed [default: 1.0]"
    echo "  --results-dir DIR         Specify results save directory [default: results]"
    echo "  --seed SEED               Specify fixed seed value"
    echo "  --no-sequential-seeds     Disable sequential seed mode (0,1,2,...)"
    echo "  -i, --interactive         Enable interactive difficulty selection"
    echo "  --headless                Enable headless mode (server environment)"
    echo "  -h, --help                Show this help information"
    echo ""
    echo "Seed configuration description:"
    echo "  By default uses sequential seeds (0, 1, 2, ..., rounds-1)"
    echo "  Use --seed to specify a fixed seed value"
    echo "  Use --no-sequential-seeds to disable sequential mode and use random seeds"
    echo ""
    echo "Examples:"
    echo "  $0 -m openai -d medium -r 3"
    echo "  $0 --model baichuan --interactive --seed 42"
    echo "  $0 --headless -m openai -d hard --no-sequential-seeds"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_TYPE="$2"
            shift 2
            ;;
        -d|--difficulty)
            DIFFICULTY="$2"
            shift 2
            ;;
        -r|--rounds)
            ROUNDS="$2"
            shift 2
            ;;
        -s|--max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --speed)
            SPEED="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --no-sequential-seeds)
            SEQUENTIAL_SEEDS=false
            shift
            ;;
        -i|--interactive)
            INTERACTIVE=true
            shift
            ;;
        --headless)
            HEADLESS=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate model type
if [[ "$MODEL_TYPE" != "openai" && "$MODEL_TYPE" != "baichuan" ]]; then
    echo "Error: Unsupported model type '$MODEL_TYPE'"
    echo "Supported model types: openai, baichuan"
    exit 1
fi

# Validate difficulty
if [[ "$DIFFICULTY" != "easy" && "$DIFFICULTY" != "medium" && "$DIFFICULTY" != "hard" ]]; then
    echo "Error: Unsupported difficulty level '$DIFFICULTY'"
    echo "Supported difficulty levels: easy, medium, hard"
    exit 1
fi

# Set headless mode
if [[ "$HEADLESS" == "true" ]]; then
    export SDL_AUDIODRIVER=dummy
    export SDL_VIDEODRIVER=dummy
    echo "Headless mode enabled"
fi

# Auto-detect Python command
PYTHON_CMD=""
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: Python interpreter not found (python or python3)"
    exit 1
fi

echo "Using Python command: $PYTHON_CMD"

# Check required Python packages
echo "Checking Python dependencies..."
echo "Using Python: $(which $PYTHON_CMD)"
echo "Python version: $($PYTHON_CMD --version)"

# Improved dependency check - check each package individually
MISSING_PACKAGES=()

# Check each required package
check_package() {
    local package=$1
    local import_name=$2
    
    if ! $PYTHON_CMD -c "import $import_name" 2>/dev/null; then
        MISSING_PACKAGES+=("$package")
        echo "❌ Missing package: $package"
        return 1
    else
        echo "✅ Installed: $package"
        return 0
    fi
}

echo "Checking individual dependency packages..."
check_package "gym" "gym"
check_package "numpy" "numpy"
check_package "pygame" "pygame"
check_package "requests" "requests"
check_package "pillow" "PIL"

# If there are missing packages, show error and exit
if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo ""
    echo "Error: Missing the following Python packages: ${MISSING_PACKAGES[*]}"
    echo "Please install in the current Python environment:"
    echo "pip install ${MISSING_PACKAGES[*]}"
    echo ""
    echo "If you are using a virtual environment, make sure the correct environment is activated"
    exit 1
else
    echo "✅ All dependency packages check passed"
fi

# Check configuration file
if [[ ! -f "config.py" ]]; then
    echo "Error: Configuration file config.py not found"
    exit 1
fi

# Check test file
TEST_FILE="test_${MODEL_TYPE}.py"
if [[ ! -f "$TEST_FILE" ]]; then
    echo "Error: Test file $TEST_FILE not found"
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "=================================="
echo "Maze Navigation Test Startup"
echo "=================================="
echo "Model type: $MODEL_TYPE"
echo "Difficulty level: $DIFFICULTY"
echo "Test rounds: $ROUNDS"
echo "Maximum steps: $MAX_STEPS"
echo "Run speed: $SPEED"
echo "Results directory: $RESULTS_DIR"
echo "Interactive mode: $INTERACTIVE"
echo "Headless mode: $HEADLESS"
if [[ -n "$SEED" ]]; then
    echo "Fixed seed: $SEED"
else
    echo "Sequential seeds: $SEQUENTIAL_SEEDS"
fi
echo "=================================="

# Build Python command arguments
PYTHON_ARGS=(
    --difficulty "$DIFFICULTY"
    --rounds "$ROUNDS" 
    --max-steps "$MAX_STEPS"
    --speed "$SPEED"
    --results-dir "$RESULTS_DIR"
)

if [[ "$INTERACTIVE" == "true" ]]; then
    PYTHON_ARGS+=(--interactive)
fi

if [[ -n "$SEED" ]]; then
    PYTHON_ARGS+=(--seed "$SEED")
fi

if [[ "$SEQUENTIAL_SEEDS" == "false" ]]; then
    PYTHON_ARGS+=(--no-sequential-seeds)
fi

# Run test
echo "Starting test execution..."
echo "Executing command: $PYTHON_CMD $TEST_FILE ${PYTHON_ARGS[*]}"
echo ""

$PYTHON_CMD "$TEST_FILE" "${PYTHON_ARGS[@]}"

echo ""
echo "Test completed! Results saved to $RESULTS_DIR directory"
