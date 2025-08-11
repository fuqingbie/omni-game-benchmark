# Bomberman AI Game

An AI-powered Bomberman game where multiple AI models compete against each other in a classic Bomberman environment. The game supports multimodal AI players that can process visual and audio information to make strategic decisions.

## Features

- **Multi-model Support**: Up to 4 different AI models can compete simultaneously
- **Multimodal Input**: AI players receive both visual (game screenshots) and audio (game events) information
- **Difficulty Levels**: Easy and Normal difficulty with different map sizes and game parameters
- **Statistics Tracking**: Comprehensive statistics tracking for each model's performance
- **Player Shuffling**: Random player position assignment across episodes to ensure fair competition

## Requirements

- Python 3.8+
- Required packages: `requests`, `numpy`, `pygame`, `opencv-python`
- AI API access (OpenAI-compatible endpoints)

## Installation

1. Clone or download the game files to your local directory
2. Install required dependencies:
```bash
pip install requests numpy pygame opencv-python
```

## Configuration

### Model Configuration File

Create a JSON configuration file (e.g., `model_config_example.json`) with your AI models:

```json
[
    {
        "api_base": "https://api.openai.com/v1",
        "api_key": "your_api_key_1",
        "model": "gpt-4-vision-preview",
        "description": "GPT-4 Vision Model"
    },
    {
        "api_base": "https://api.anthropic.com/v1", 
        "api_key": "your_api_key_2",
        "model": "claude-3-opus",
        "description": "Claude 3 Opus"
    },
    {
        "api_base": "https://api.example.com/v1",
        "api_key": "your_api_key_3", 
        "model": "custom-model",
        "description": "Custom Model"
    }
]
```

**Note**: Do not include `player_id` in the configuration file. Player positions are automatically shuffled between episodes for fair competition.

### Difficulty Levels

The game supports two difficulty levels:

#### Easy Mode
- **Map Size**: 9x7 grid (smaller map)
- **Soft Wall Probability**: 0.3 (fewer obstacles)
- **Clear Radius**: 2 (larger starting area)
- **Player Speed**: 2 (faster movement)
- **Max Move Distance**: 6 tiles
- **Item Drop Rate**: 0.5 (higher power-up drop rate)

#### Normal Mode (Default)
- **Map Size**: 13x11 grid (standard map)
- **Soft Wall Probability**: 0.6 (more obstacles)
- **Clear Radius**: 1 (standard starting area)
- **Player Speed**: 1 (standard movement)
- **Max Move Distance**: 5 tiles
- **Item Drop Rate**: 0.2 (standard power-up drop rate)

## Usage

### Quick Start

```bash
python start_ai_game.py --config model_config_example.json
```

### Advanced Usage

```bash
python start_ai_game.py \
    --config model_config_example.json \
    --episodes 5 \
    --steps 500 \
    --delay 0.3 \
    --difficulty normal
```

### Command Line Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--config` | string | `model_config_example.json` | Path to model configuration file |
| `--episodes` | int | 3 | Number of game episodes to play |
| `--steps` | int | 300 | Maximum steps per episode |
| `--delay` | float | 0.5 | Delay between steps (seconds) |
| `--difficulty` | string | `easy` | Game difficulty (`easy` or `normal`) |

### Direct Execution

You can also run the multi-model game directly:

```bash
python multi_model_game.py \
    --config model_config_example.json \
    --episodes 3 \
    --steps 300 \
    --delay 0.3 \
    --difficulty easy
```

## Game Rules

1. **Objective**: Eliminate all other players to win
2. **Actions**: Players can move (up to max distance) or place bombs
3. **Bombs**: Create cross-shaped explosions that destroy soft walls and eliminate players
4. **Power-ups**: Destroying soft walls may drop items that enhance abilities:
   - **Fire Power**: Increases bomb explosion range
   - **Bomb Count**: Allows placing more bombs simultaneously
   - **Speed**: Increases movement speed and distance
5. **Victory**: Last surviving player wins the episode

## Output and Statistics

### Real-time Output
- Player decisions and actions
- Game events (movements, bomb placements, explosions)
- Episode results and winner announcements

### Saved Results
Results are automatically saved to the `result/` directory:

- **Episode Results**: `bomberman_episode_X_YYYYMMDD_HHMMSS.json`
- **Final Statistics**: `bomberman_final_stats_YYYYMMDD_HHMMSS.json`

### Statistics Tracked
- Total kills per model
- Total deaths per model
- Items collected per model
- Win count and win rate
- Episode-by-episode performance

## ⚠️ IMPORTANT WARNINGS

### DO NOT RUN MULTIPLE INSTANCES SIMULTANEOUSLY

**Critical**: Do not run multiple game instances at the same time. This will cause:
- **File conflicts**: Multiple processes writing to the same result files
- **Data corruption**: Statistics and results may be overwritten or corrupted
- **Resource conflicts**: GPU/API rate limiting issues

**Best Practice**: Run games sequentially, one at a time.

### API Rate Limiting
- Monitor your API usage to avoid rate limits
- Consider increasing `--delay` if you encounter rate limiting
- Different AI providers have different rate limits

### Resource Usage
- The game requires significant computational resources for AI inference
- Visual rendering may consume additional GPU resources
- Audio processing requires memory for multimodal inputs

## Troubleshooting

### Common Issues

1. **Configuration file not found**
   ```
   Error: Configuration file model_config_example.json does not exist
   ```
   Solution: Create the configuration file with valid API credentials

2. **API authentication errors**
   ```
   HTTP 401: Unauthorized
   ```
   Solution: Check your API keys and endpoints in the configuration file

3. **Model not responding**
   ```
   Player X decision error: timeout
   ```
   Solution: Increase timeout values or check API connectivity

4. **JSON parsing errors**
   ```
   JSON decode failed, returning default action
   ```
   Solution: This is handled gracefully with fallback actions

### Performance Optimization

- Use `--delay 0.1` for faster games (minimum recommended)
- Reduce `--steps` for shorter episodes
- Use `easy` difficulty for faster games with smaller maps

## Examples

### Quick 3-episode game on easy difficulty:
```bash
python start_ai_game.py --difficulty easy --episodes 3 --delay 0.2
```

### Extended 10-episode tournament:
```bash
python start_ai_game.py --episodes 10 --steps 500 --difficulty normal --delay 0.5
```

### Fast testing mode:
```bash
python start_ai_game.py --episodes 1 --steps 100 --delay 0.1 --difficulty easy
```

## Support

If you encounter issues:
1. Check the console output for error messages
2. Verify your configuration file format
3. Ensure all API credentials are valid
4. Check that no other instances are running simultaneously

Remember: **Never run multiple instances at the same time** to avoid data corruption and file conflicts.
