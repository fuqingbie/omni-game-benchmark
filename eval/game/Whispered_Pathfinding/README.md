# Maze Navigation Test Tool

This is a tool for testing the performance of large language models in maze navigation tasks, supporting both OpenAI and Baichuan models.

## File Structure

```
.
├── config.py                  # Configuration file
├── test_openai.py            # OpenAI model test
├── test_baichuan.py          # Baichuan model test  
├── run_test.sh               # Launch script
├── analyze_maze_results.py   # Results analysis tool
├── maze_gym_env.py           # Maze environment
└── results/                  # Test results directory
```

## Configuration

Before use, please modify the configuration in `config.py`:

### API Configuration
```python
# OpenAI-format API
API_BASE = "http://your-api-server/v1"
API_KEY = "your-api-key"
MODEL_CHAT = "your-model-name"

# Baichuan FastAPI
BAICHUAN_FASTAPI_BASE_URL = "http://your-baichuan-server"
```

### Environment Configuration
```python
# Headless mode - Set to True for server environments
HEADLESS_MODE = False
```

## Usage

### 1. Using Launch Script (Recommended)

```bash
# Grant execution permission
chmod +x run_test.sh

# Basic usage
./run_test.sh -m openai -d easy -r 5

# Interactive difficulty selection
./run_test.sh -m baichuan --interactive

# Headless mode (server environment)
./run_test.sh --headless -m openai -d hard

# With seed configuration
./run_test.sh -m openai -d medium --seed 42
./run_test.sh -m openai -d hard --no-sequential-seeds
```

### 2. Direct Python Execution

```bash
# OpenAI model test
python test_openai.py --difficulty medium --rounds 3

# Baichuan model test  
python test_baichuan.py --interactive

# With seed control
python test_openai.py --seed 123 --rounds 5
```

### 3. Results Analysis

After running tests, analyze the results using the built-in analysis tool:

```bash
# Basic analysis
python analyze_maze_results.py

# Specify results directory
python analyze_maze_results.py --results-dir my_results

# Export data to JSON format
python analyze_maze_results.py --export-json

# Custom analysis
python analyze_maze_results.py --results-dir results --export-json
```

## Parameter Description

### Test Parameters
- `-m, --model`: Model type (openai|baichuan)
- `-d, --difficulty`: Difficulty level (easy|medium|hard)
- `-r, --rounds`: Number of test rounds
- `-s, --max-steps`: Maximum steps per round
- `--speed`: Auto-run speed
- `--results-dir`: Results save directory
- `--seed`: Fixed seed value for reproducibility
- `--no-sequential-seeds`: Disable sequential seed mode (0,1,2,...)
- `-i, --interactive`: Interactive difficulty selection
- `--headless`: Headless mode

### Analysis Parameters
- `--results-dir`: Directory containing result files
- `--export-json`: Export analysis data to JSON format
- `--format`: Output format (text|json)

## Seed Configuration

The tool supports three seed modes:

1. **Sequential Seeds (Default)**: Uses seeds 0, 1, 2, ... for each round
2. **Fixed Seed**: Use `--seed N` to set a fixed seed for all rounds
3. **Random Seeds**: Use `--no-sequential-seeds` for random seeds

This ensures reproducible experiments for research purposes.

## System Requirements

- Python 3.7+
- Dependencies: gym, numpy, pygame, requests, pillow
- Graphics interface (optional, can use headless mode for server environments)
- Game assets directory: `../assets-necessay/` (automatically detected)

## Installation

```bash
pip install gym numpy pygame requests pillow
```

## Results

### Result Files
Test results are saved in the `results/` directory with filename format:
`{script_name}_round{round_number}_{timestamp}.txt`

Each result file contains:
- Difficulty level
- Seed value used
- Total steps taken
- Total reward
- Number of invalid actions
- Success status
- Truncation status
- Timestamp

### Analysis Reports
The analysis tool generates comprehensive reports including:

#### Overall Statistics
- Total number of games
- Overall success rate
- Truncation rate
- Step statistics (average, min, max, median)
- Reward statistics
- Invalid action statistics

#### Model Comparison
- Performance comparison between different models
- Success rates by model type
- Average steps and rewards by model

#### Difficulty Analysis
- Performance across different difficulty levels
- Success rates for easy/medium/hard modes
- Step requirements by difficulty

#### Cross Analysis
- Model performance on specific difficulty levels
- Detailed breakdown by model × difficulty combinations

### Sample Analysis Output
```
============================================================
Maze Navigation Analysis Report
============================================================
Total files analyzed: 15
Analysis timestamp: 2024-01-20 15:30:45

OVERALL STATISTICS:
  Total games: 15
  Success rate: 80.0% (12/15)
  Truncated rate: 6.7% (1/15)
  Average steps: 127.3
  Step range: 45 - 389
  Median steps: 98
  Average reward: 0.85
  Unique seeds used: 15
```

## Example Workflows

### Basic Testing Workflow
```bash
# 1. Configure API settings in config.py
# 2. Run tests
./run_test.sh -m openai -d easy -r 5 --seed 42

# 3. Analyze results
python analyze_maze_results.py --export-json
```

### Comprehensive Evaluation
```bash
# Test multiple difficulties
./run_test.sh -m openai -d easy -r 10
./run_test.sh -m openai -d medium -r 10  
./run_test.sh -m openai -d hard -r 10

# Compare with another model
./run_test.sh -m baichuan -d easy -r 10
./run_test.sh -m baichuan -d medium -r 10
./run_test.sh -m baichuan -d hard -r 10

# Generate comprehensive analysis
python analyze_maze_results.py --export-json
```

### Reproducible Research
```bash
# Use fixed seeds for reproducibility
for seed in {0..9}; do
    ./run_test.sh -m openai -d medium --seed $seed -r 1
done

# Analyze reproducible results
python analyze_maze_results.py --results-dir results
```

## Troubleshooting

### Assets Path Issues
If you encounter `FileNotFoundError` for audio assets:
- Ensure the `assets-necessay` directory exists in the parent directory
- The tool will automatically search for assets in multiple locations
- Audio features may be disabled if assets are not found

### Python Command Issues
- The script automatically detects `python` or `python3` commands
- Ensure you're using the correct Python environment with all dependencies installed

### Headless Mode
For server environments without display:
- Use `--headless` flag or set `HEADLESS_MODE = True` in config.py
- This disables SDL graphics and audio drivers

### API Connection Issues
- Verify API endpoints are accessible
- Check API keys and authentication
- Ensure models support multimodal inputs (text + image + audio)

### Memory and Performance
- Large numbers of rounds may consume significant memory
- Use smaller batch sizes for limited memory environments
- Monitor disk space for result files

## Contributing

When contributing to this project:
1. Follow the existing code structure
2. Update configuration options in `config.py`
3. Ensure compatibility with both headless and GUI modes
4. Test with multiple models and difficulty levels
5. Update this README with new features

## License

[Add license information here]

## Citation

If you use this tool in your research, please cite:
```
[Add citation information here]
```
