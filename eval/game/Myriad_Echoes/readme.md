# Rhythm Memory Game AI Evaluation

This project evaluates AI model performance on a rhythm memory game. It supports multiple AI models including Baichuan, OpenAI GPT, and other compatible models through their respective APIs.

## Overview

The rhythm memory game tests an AI's ability to:
1. Watch and listen to audiovisual sequences
2. Memorize the order of icons lighting up
3. Reproduce the sequence by clicking icons in the correct order

## Prerequisites

- Python 3.8+
- Pygame
- PyTorch/TensorFlow (for model dependencies)
- FFmpeg (for video processing)
- Access to AI model APIs (Baichuan FastAPI server, OpenAI API, etc.)

## Installation

```bash
pip install numpy pygame pillow requests moviepy opencv-python pathlib
```

## Supported Models

### 1. Baichuan Model (FastAPI)
- File: `eval_baichuan_multi_episode.py`
- Requires: Baichuan FastAPI server
- Supports: Video, Audio, and Image inputs

### 2. OpenAI GPT Models
- File: `eval_openai_multi_episode.py`
- Requires: OpenAI API key
- Supports: Video (as frames), Audio, and Image inputs

### 3. OpenAI GPT Models (Frame-based)
- File: `eval_openai_video_frame_multi_episode.py`
- Requires: OpenAI API key  
- Supports: Video frames (0.5s intervals), Audio, and Image inputs

## Configuration

### Baichuan FastAPI Server

```python
# In eval_baichuan_multi_episode.py, line 11
FASTAPI_BASE_URL = "http://your-server:port"  # Replace with your actual server URL
```

### OpenAI API Configuration

```python
# In eval_openai_multi_episode.py or eval_openai_video_frame_multi_episode.py
API_BASE = "https://api.openai.com/v1"  # Or your custom endpoint
API_KEY = "your-openai-api-key"
MODEL_CHAT = "gpt-4o"  # Or your preferred model
```

### Game Difficulty Levels

The game supports three difficulty levels:

1. **Easy (difficulty=1)**: 6 icons, 2 rows × 3 columns
2. **Normal (difficulty=2)**: 10 icons, 2 rows × 5 columns  
3. **Hard (difficulty=3)**: 15 icons, 3 rows × 5 columns

## Usage

### Basic Usage

For Baichuan model:
```bash
python eval_baichuan_multi_episode.py
```

For OpenAI models:
```bash
python eval_openai_multi_episode.py
```

For OpenAI frame-based evaluation:
```bash
python eval_openai_video_frame_multi_episode.py
```

### Customizing Parameters

Edit the following parameters in the `__main__` section of any evaluation script:

```python
if __name__ == "__main__":
    difficulty = 1        # Set difficulty level (1-3)
    max_episodes = 10     # Set number of episodes to run
    runner = ModelRhythmMemoryRunner(difficulty=difficulty, max_episodes=max_episodes)
    runner.run()
```

### Command Line Parameters

You can modify any script to accept command line arguments:

```python
import argparse

parser = argparse.ArgumentParser(description='Rhythm Memory Game AI Evaluation')
parser.add_argument('--difficulty', type=int, default=1, choices=[1,2,3], 
                   help='Game difficulty: 1=Easy, 2=Normal, 3=Hard')
parser.add_argument('--episodes', type=int, default=10, 
                   help='Number of episodes to run')
parser.add_argument('--model', type=str, default="baichuan", 
                   choices=["baichuan", "openai", "openai-frames"],
                   help='Model type to evaluate')

args = parser.parse_args()
```

## Output Files

All evaluation scripts generate consistent output files:

### Data Directory Structure
```
ai_data/
└── rhythm_memory_ai_YYYYMMDD_HHMMSS/
    ├── model_inputs/           # All model input data
    │   ├── input_001_sequence_analysis/
    │   ├── input_002_click_suggestion_step_2/
    │   └── ...
    ├── screen_capture.jpg      # Game interface screenshots
    └── rhythm_memory_results/  # Evaluation results
        ├── eval_[model]_multi_episode_detailed_results_TIMESTAMP.json
        └── eval_[model]_multi_episode_summary_TIMESTAMP.txt
```

### Result Files

1. **Detailed JSON Results**: Complete episode data with predictions and accuracy
2. **Summary Text Report**: Overall statistics and success rates
3. **Model Input Archive**: All data sent to the model for debugging

## Important Warnings

### ⚠️ DO NOT RUN MULTIPLE INSTANCES SIMULTANEOUSLY

**CRITICAL**: Only run one instance of any evaluation script at a time. Running multiple instances in parallel will cause:

- **File Overwrite Issues**: Temporary files and results may be overwritten
- **Resource Conflicts**: Pygame and audio resources cannot be shared
- **Data Corruption**: Model input archives may become corrupted
- **Inaccurate Results**: Statistics and scores may be incorrectly calculated

### Recommended Workflow

1. **Choose One Model**: Select one evaluation script at a time
2. **Run Sequential Tests**: Complete one difficulty level before starting the next
3. **Wait for Completion**: Ensure each evaluation finishes before running another
4. **Check Results**: Verify result files before starting additional tests
5. **Model Comparison**: Use different output directories for different models

## Model-Specific Features

### Baichuan Model
- **Session Management**: Automatic session clearing between episodes
- **Error Handling**: Robust connection and timeout handling
- **Input Formats**: Direct video, audio, and image file uploads

### OpenAI Models
- **Video Processing**: 
  - Standard: Direct video upload (if supported)
  - Frame-based: Extracts frames every 0.5 seconds
- **Audio Support**: Base64 encoded audio input
- **Token Management**: Efficient prompt and response handling

## Monitoring Progress

All scripts provide real-time progress information in English:

```
===== Episode 1/10 =====
--- Sequence Playback Phase ---
Starting sequence analysis...
--- Sequence Reproduction Phase ---
Progress: 1/10, Current Success Rate: 100.0%
```

Progress is automatically saved every 10 episodes to prevent data loss.

## Troubleshooting

### Common Issues

1. **API Connection Failed**
   ```
   ❌ Unable to connect to [Model] API: Connection refused
   ```
   - Check if the API server is running
   - Verify API endpoint URL and credentials
   - Ensure network connectivity

2. **Video/Audio Files Not Found**
   ```
   Sequence analysis failed
   ```
   - Ensure FFmpeg is properly installed
   - Check file permissions in the working directory
   - Verify video/audio generation is working

3. **Pygame Display Issues**
   ```
   pygame.error: No available video device
   ```
   - Run in a graphical environment
   - Set DISPLAY environment variable if using SSH
   - Install appropriate graphics drivers

### Model-Specific Issues

**Baichuan Model:**
- Session timeout: Increase timeout values in API calls
- FastAPI server overload: Reduce concurrent requests

**OpenAI Models:**
- Rate limiting: Add delays between API calls
- Token limits: Reduce frame count or audio length
- API key issues: Verify key validity and permissions

### Performance Tips

- Use SSD storage for faster video processing
- Ensure sufficient RAM (>4GB recommended)
- Close other applications that use audio/video resources
- Monitor API usage and costs for cloud-based models

## Results Interpretation

### Success Metrics

- **Episode Success Rate**: Percentage of completed sequences
- **Coordinate Accuracy**: Correct position predictions
- **Icon Recognition Accuracy**: Correct icon identification  
- **Sequence Parseability**: Model's ability to analyze sequences

### Example Output

```
===== Final Statistics =====
Total episodes: 10
Successful episodes: 7
Success rate: 70.0%
Total score: 42
Average score: 4.2
Sequence analysis errors: 1
Click prediction errors: 2
Unparseable sequences: 0
```

### Model Comparison

When comparing different models:
1. Use the same difficulty level and episode count
2. Run tests under similar conditions
3. Compare detailed JSON results for in-depth analysis
4. Consider both accuracy and error patterns

## License

This project is for research and evaluation purposes. Please ensure compliance with:
- Individual model usage policies
- API terms of service
- Data privacy regulations
- Research ethics guidelines