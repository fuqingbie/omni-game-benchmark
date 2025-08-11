# Phantom Soldiers in the Fog - Multi-Episode Evaluation

This repository contains evaluation scripts for testing AI models on a cooperative military command game using multi-modal inputs (video, audio, images). The game involves commanding a team of military units to complete objectives while dealing with hidden objectives, movement uncertainty, and strategic coordination.

## Overview

The evaluation suite includes three main scripts:

1. **`eval-oepnai-multi-episode.py`** - Direct video input evaluation
2. **`eval-oepnai-video-frame-multi-episode.py`** - Video frame extraction evaluation  
3. **`eval-baichuan-multi-episode.py`** - Baichuan model evaluation

## Key Differences Between Video Evaluation Scripts

### Direct Video Input (`eval-oepnai-multi-episode.py`)
- Sends video data directly to the API as base64-encoded video
- Supports native video processing by models that accept video modality
- Uses `modalities: ["text", "audio", "video"]` in API calls
- More efficient for models with built-in video understanding

### Video Frame Extraction (`eval-oepnai-video-frame-multi-episode.py`)
- Extracts frames from video at 0.5-second intervals using MoviePy
- Converts video sequences into multiple image inputs
- Sends image sequences instead of video to the API
- Uses `modalities: ["text", "audio"]` (no video modality)
- Compatible with models that don't support video but handle multiple images
- Requires `moviepy` and `opencv-python` dependencies

## Game Mechanics

### Objective
Command a military team to complete objectives while exploring for hidden objectives and managing team resources.

### Key Features
- **Hidden Objectives**: Some objectives are not visible initially and must be discovered through exploration
- **Movement Uncertainty**: Units don't move to exact coordinates due to role-based precision and health status
- **Multi-Modal Input**: Game state provided through video/images, audio communications, and vector data
- **Command Compliance**: Models must provide exactly ONE command per turn

### Team Roles
- **Scout**: High discovery rate (80%), low movement error, 85% command success
- **Heavy**: High movement error, 75% command success  
- **Medic**: 90% command success rate
- **Engineer**: 80% command success rate
- **Sniper**: 70% command success rate

## Installation

### Prerequisites
```bash
pip install numpy pillow requests opencv-python moviepy gym
```

### Optional Dependencies
- **MoviePy**: Required for video frame extraction script
- **OpenCV**: Required for video processing

### Environment Setup
1. Set API credentials in the script files:
   ```python
   API_BASE = "your-api-endpoint"
   API_KEY = "your-api-key" 
   MODEL_CHAT = "your-model-name"
   ```

2. For Baichuan evaluation, set:
   ```python
   FASTAPI_BASE_URL = "your-baichuan-endpoint"
   ```

## Usage

### Basic Commands

#### Direct Video Input Evaluation
```bash
python eval-oepnai-multi-episode.py --difficulty medium --num_episodes 10
```

#### Video Frame Extraction Evaluation  
```bash
python eval-oepnai-video-frame-multi-episode.py --difficulty medium --num_episodes 10
```

#### Baichuan Model Evaluation
```bash
python eval-baichuan-multi-episode.py --difficulty medium --num_episodes 10
```

### Command Line Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--difficulty` | str | `"medium"` | Game difficulty: `normal`, `medium`, `hard` |
| `--seed_index` | int | `0` | Base seed for reproducible results |
| `--max_rounds` | int | `100` | Maximum rounds per episode |
| `--num_episodes` | int | `10` | Number of episodes to evaluate |
| `--save_media` | flag | `True` | Save media files (images, videos, audio) |
| `--probabilistic_commands` | flag | `False` | Enable probabilistic command execution |
| `--api_provider` | str | `"qwen"` | API provider: `qwen`, `openai` (not for Baichuan) |
| `--model_name` | str | `None` | Override default model name |
| `--input_mode` | str | `"video"` | Input mode: `video`, `image_audio` |
| `--no_vector_text` | flag | `False` | Exclude vector info from text prompt |
| `--enhanced_video` | flag | `False` | Enable enhanced video with integrated audio |
| `--video_fps` | float | `0.5` | Video recording frame rate |
| `--audio_duration_per_frame` | float | `3.0` | Audio duration per video frame (seconds) |
| `--no_stream` | flag | `False` | Disable streaming responses |

### Example Commands

#### Standard Evaluation
```bash
# Medium difficulty, 5 episodes, save media
python eval-oepnai-multi-episode.py \
    --difficulty medium \
    --num_episodes 5 \
    --max_rounds 100 \
    --save_media
```

#### High Difficulty Evaluation
```bash
# Hard difficulty with probabilistic commands
python eval-oepnai-video-frame-multi-episode.py \
    --difficulty hard \
    --num_episodes 3 \
    --probabilistic_commands \
    --enhanced_video
```

#### Custom Model Evaluation
```bash
# Custom model with specific settings
python eval-oepnai-multi-episode.py \
    --model_name "gpt-4o" \
    --difficulty normal \
    --num_episodes 10 \
    --no_vector_text \
    --video_fps 1.0
```

#### Baichuan Evaluation
```bash
# Baichuan model evaluation
python eval-baichuan-multi-episode.py \
    --difficulty medium \
    --num_episodes 5 \
    --enhanced_video \
    --video_fps 0.5
```

## Difficulty Levels

### Normal
- Basic objectives and team coordination
- Moderate hidden objective density
- Standard movement uncertainty

### Medium  
- Increased objective complexity
- More hidden objectives to discover
- Higher movement uncertainty
- Additional strategic challenges

### Hard
- Complex multi-stage objectives
- Maximum hidden objective density
- Highest movement uncertainty
- Advanced tactical scenarios

## Output Structure

### Directory Organization
```
outputs/
├── {provider}_eval_{difficulty}_seed{seed}_{episodes}ep_{timestamp}/
│   ├── episode_00/
│   │   ├── images/          # Frame-by-frame screenshots
│   │   ├── videos/          # Video recordings
│   │   ├── audio/           # Audio communications
│   │   └── responses/       # Model responses
│   ├── episode_01/
│   │   └── ...
│   └── results.json         # Comprehensive results
```

### Results Format
```json
{
  "config": {
    "difficulty": "medium",
    "num_episodes": 10,
    "api_provider": "qwen",
    "model": "model-name"
  },
  "episodes": [
    {
      "episode_index": 0,
      "final_stats": {
        "final_score_normalized": 85.2,
        "objectives_completed": 8,
        "total_objectives": 10,
        "success_rate": 0.8
      }
    }
  ],
  "summary_stats": {
    "score_stats": {
      "mean": 82.5,
      "std": 12.3,
      "min": 65.0,
      "max": 95.0
    },
    "objectives_stats": {
      "mean_success_rate": 0.75,
      "episodes_with_100_percent": 3
    }
  }
}
```

## Command Format

Models must provide exactly ONE command per turn:

### Command Types
- **Individual**: `COMMAND: [member_id] [action] [x] [y]`
- **Team**: `COMMAND: all [action] [x] [y]`  
- **Multi-member**: `COMMAND: 0,1,2 [action] [x] [y]`

### Actions
- `move`: Move to coordinates
- `attack`: Attack at location
- `defend`: Set defensive position
- `recon`: Reconnaissance mission
- `status`: Check unit status

### Examples
```
COMMAND: 0 recon 25 30        # Scout reconnaissance  
COMMAND: all move 45 20       # Team movement
COMMAND: 0,1 attack 70 80     # Multi-unit attack
```

## Evaluation Metrics

### Performance Metrics
- **Score**: Normalized mission score (0-100)
- **Success Rate**: Percentage of objectives completed
- **Efficiency**: Steps taken per objective
- **Consistency**: Performance variation across episodes

### Command Compliance
- **Valid Single Commands**: Properly formatted single commands
- **Multiple Command Violations**: Attempts to issue multiple commands
- **No Command Found**: Responses without valid commands
- **Compliance Rate**: Overall command format adherence

## Troubleshooting

### Common Issues

1. **MoviePy Import Error** (frame extraction script):
   ```bash
   pip install moviepy
   ```

2. **API Connection Failed**:
   - Verify API credentials and endpoints
   - Check network connectivity
   - Ensure model supports required modalities

3. **Video Processing Error**:
   - Ensure video files are in MP4 format
   - Check video encoding compatibility
   - Verify sufficient disk space

4. **Memory Issues**:
   - Reduce `--num_episodes` for large evaluations
   - Use `--no_save_media` to reduce storage
   - Monitor system resources during evaluation

### Performance Tips

1. **For Large-Scale Evaluation**:
   - Use `--no_save_media` to save disk space
   - Reduce `--video_fps` for smaller video files
   - Run episodes sequentially rather than parallel

2. **For Model Comparison**:
   - Use identical `--seed_index` for reproducible results
   - Keep `--max_rounds` consistent across evaluations
   - Use same difficulty and episode count

## API Integration

### Supported Providers
- **Qwen**: Full multi-modal support (video, audio, images)
- **OpenAI**: Vision and audio support (varies by model)
- **Baichuan**: Custom FastAPI integration with file uploads

### Adding New Providers
1. Extend `API_CONFIGS` dictionary
2. Implement provider-specific query methods
3. Update message building for provider API format
4. Test with provider's modality support

## Contributing

When adding new features:
1. Maintain backward compatibility with existing command line arguments
2. Update this README with new parameters
3. Test with multiple difficulty levels and episode counts
4. Ensure proper error handling and logging

## License

This evaluation suite is designed for research and development of multi-modal AI systems in strategic gaming environments.
