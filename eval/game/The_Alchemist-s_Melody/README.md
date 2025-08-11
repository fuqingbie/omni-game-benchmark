# The Alchemist's Melody - Musical AI Agent Game

A multimodal AI-powered musical sequence learning game where agents learn correct musical patterns through auditory and visual feedback.

## 🎵 Project Overview

The Alchemist's Melody is an innovative AI training environment that combines:
- **Musical Theory**: Sequence learning based on do-re-mi scales
- **Multimodal Perception**: Image recognition + audio analysis
- **Reinforcement Learning**: Strategy optimization through feedback
- **Memory Management**: Intelligent conversation history compression and retrieval

### Core Game Mechanics
- Agents play musical notes by clicking different colored blocks
- Color-to-note mapping is **randomized** each round
- Goal is to play the correct musical sequence (do→re→mi→fa→sol→la→si)
- Any incorrect click resets the sequence to the beginning

## 🚀 Quick Start

### Requirements
```bash
Python >= 3.8
pygame >= 2.0.0
numpy >= 1.20.0
pillow >= 8.0.0
requests >= 2.25.0
gymnasium >= 0.26.0
```

### Install Dependencies
```bash
pip install pygame numpy pillow requests gymnasium sounddevice soundfile librosa
```

### Configure API
1. Edit the `config.py` file:
```python
class API:
    BASE_URL = "your_api_base_url"  # Enter your API base URL
    API_KEY = "your_api_key"        # Enter your API key
    MODEL_CHAT = "gemini-pro-2.5"   # Or your preferred model
```

2. Or use environment variables:
```bash
export ALCHEMIST_API_BASE="your_api_base_url"
export ALCHEMIST_API_KEY="your_api_key"
export ALCHEMIST_MODEL="gemini-pro-2.5"
```

### Run the Game
```bash
# Run multimodal agent
python run_mm_agent.py

# Run Baichuan model agent
python run_baichuan_agent.py

# Run game directly (manual mode)
python sound_alchemist_game.py
```

## 🎮 Game Rules

### Basic Rules
1. **Musical Sequence**: Strictly follow ascending order do→re→mi→fa→sol→la→si
2. **Random Mapping**: Color-to-note correspondence is randomized at the start of each round
3. **Sequence Reset**: Any error resets progress to the first note of the current round
4. **Learning Mechanism**: Agents must learn color-note mappings through audio feedback

### Difficulty Levels
- **Easy**: 3-note sequence (e.g., do→re→mi)
- **Normal**: 5-note sequence (e.g., re→mi→fa→sol→la)
- **Hard**: 7-note sequence (e.g., do→re→mi→fa→sol→la→si)

### Scoring System
- **Base Score**: 1000 points
- **Difficulty Multiplier**: Easy×1, Normal×2, Hard×3
- **Sequence Bonus**: 50 points per note
- **Perfect Play Bonus**: 500 points
- **Error Penalty**: -150 points per mistake

## 🤖 Agent Architecture

### Multimodal Perception
```python
observation = {
    "image": np.ndarray,    # 224×224×3 game screen
    "audio": np.ndarray,    # 16000Hz×1s audio data  
    "state": np.ndarray     # [score, lives, solved, tick]
}
```

### Decision Process
1. **Observation Analysis**: Parse colored blocks and game state from images
2. **Audio Processing**: Analyze audio feedback from previous actions
3. **Memory Retrieval**: Extract relevant experience from conversation history
4. **Strategic Reasoning**: Make decisions based on current state and historical experience
5. **Action Execution**: Select color block to click

### Memory Management System
- **Native Memory**: Maintain last 8 rounds of complete conversations
- **Compressed Summaries**: Key information summaries from historical rounds
- **RAG Retrieval**: Retrieve historical experience based on relevance
- **Hybrid Strategy**: Combine native memory and RAG retrieval

## 📊 Data Saving

### Auto-saved Content
- **Image Data**: Game screenshots for each step (PNG format)
- **Audio Data**: Audio feedback for each step (WAV format, 16kHz)
- **Game State**: Detailed JSON metadata
- **Sequence Data**: Complete episode step sequences
- **Scoring Data**: Performance evaluation and learning progress

### Data Directory Structure
```
game_data/caclu/
├── images/          # Game screenshots
├── audio/           # Audio files
├── metadata/        # Game state JSON
├── sequences/       # Episode sequence data
└── scores/          # Scoring and statistics
```

## 🛠️ Development Guide

### Project Structure
```
The_Alchemist-s_Melody/
├── config.py                    # Configuration file
├── sound_alchemist_game.py      # Core game logic
├── sound_alchemist_env.py       # Gymnasium environment wrapper
├── multimodal_agent.py          # Main agent implementation
├── multimodal_agent_baichuan.py # Baichuan model agent
├── run_mm_agent.py              # Agent execution script
└── assets/                      # Game asset files
```

### Core Class Descriptions

#### `SoundAlchemistEnv`
Standard Gymnasium environment providing:
- `reset()`: Reset game state
- `step(action)`: Execute action and return observation
- `render()`: Render game screen

#### `MultimodalAgent`
Multimodal agent with features:
- Image + audio multimodal perception
- LLM reasoning and decision making
- Memory management and experience replay
- Local fallback mechanism

#### `ConversationMemoryManager`
Conversation memory manager:
- Native conversation history maintenance
- Automatic compression of old rounds
- Relevance-based experience retrieval

### Extension Development

#### Adding New Agents
```python
class CustomAgent:
    def __init__(self, **kwargs):
        # Initialize agent
        pass
    
    def act(self, obs: dict) -> tuple:
        # Implement decision logic
        action_id = self.choose_action(obs)
        end_game = self.check_termination(obs)
        return action_id, end_game
```

#### Custom Scoring System
```python
def custom_scoring_function(state, action, reward):
    # Implement custom scoring logic
    custom_score = calculate_custom_metrics(state, action)
    return custom_score
```

## 📈 Performance Monitoring

### Key Metrics
- **Completion Rate**: Percentage of successfully completed sequences
- **Average Steps**: Average steps required to complete sequences
- **Learning Curve**: Performance improvement over time
- **Error Analysis**: Common error patterns and causes

### Real-time Monitoring
```python
# Get agent statistics
stats = agent.get_model_output_stats()
print(f"Successful calls: {stats['successful_calls']}/{stats['total_calls']}")
print(f"Average response time: {stats['avg_response_time']:.2f}s")

# Get game progress
progress = agent.get_learning_progress_summary()
print(progress)
```

## 🔧 Troubleshooting

### Common Issues

#### 1. API Connection Failure
```
Error: Connection error / Timeout
Solution: Check API_BASE and API_KEY configuration in config.py
```

#### 2. Audio Device Issues
```
Error: sounddevice error
Solution: Install audio dependencies or disable audio capture in config
```

#### 3. Game Window Black Screen
```
Error: Game screen not displaying
Solution: Check pygame installation and display drivers
```

#### 4. High Memory Usage
```
Error: Memory usage continuously growing
Solution: Adjust MAX_NATIVE_HISTORY and MAX_TOTAL_MEMORY in config
```

### Debug Mode
```python
# Enable verbose output
Config.Agent.VERBOSE = True
Config.Debug.LOG_LEVEL = "DEBUG"

# Save debug data
Config.Debug.SAVE_API_REQUESTS = True
Config.Debug.SAVE_GAME_STATES = True
```

## 🎯 Experiment Suggestions

### Training Strategies
1. **Gradual Difficulty Increase**: Start with Easy, gradually progress to Hard
2. **Comparative Experiments**: Test different memory management strategies
3. **Ablation Studies**: Separately test importance of visual and auditory components
4. **Multi-round Training**: Observe long-term learning effects

### Evaluation Metrics
- Episodes required for first success
- Time to achieve stable performance
- Adaptation speed across different difficulties
- Knowledge retention across sessions

## 📚 Related Resources

### Musical Theory Background
- [Scale Theory](https://en.wikipedia.org/wiki/Musical_scale)
- [Do-Re-Mi System](https://en.wikipedia.org/wiki/Solfège)

### Multimodal Learning
- [Multimodal Deep Learning](https://arxiv.org/abs/2301.04856)
- [Audio-Visual Fusion Methods](https://arxiv.org/abs/2010.09478)

### Reinforcement Learning
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Multi-Agent Environment Design](https://arxiv.org/abs/2006.07869)

## 🤝 Contributing

Welcome to submit Issues and Pull Requests!

### Development Process
1. Fork this project
2. Create feature branch
3. Commit changes
4. Create Pull Request

### Code Standards
- Use Python type hints
- Follow PEP 8 standards
- Add detailed docstrings
- Write unit tests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## 📞 Contact

- Project Maintainer: [Your Name]
- Email: [your.email@example.com]
- Project Homepage: [GitHub Link]

---

*The Alchemist's Melody - Teaching AI the Magic of Music* 🎼✨
