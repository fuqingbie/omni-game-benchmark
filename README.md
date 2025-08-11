# OmniPlay - Multimodal AI Gaming Evaluation Platform

A comprehensive platform for evaluating large language models' performance in multimodal gaming environments, supporting visual, audio, and text input modalities.

## 🎮 Supported Games

### 1. Whispered Pathfinding
- **Modalities**: Image + Audio + Vector Information
- **Task**: Navigate through maze to find targets based on voice guidance
- **Difficulty**: Easy/Medium/Hard
- **Scripts**: `test_openai.py`, `test_baichuan.py`

### 2. The Alchemist's Melody
- **Modalities**: Image + Audio + State Vector
- **Task**: Learn and reproduce musical sequence patterns
- **Features**: Random color-note mapping, requires learning capability
- **Scripts**: `run_mm_agent.py`, `run_baichuan_agent.py`

### 3. Phantom Soldiers in the Fog
- **Modalities**: Video/Image Sequence + Audio + Vector Data
- **Task**: Command military units to complete objectives
- **Features**: Hidden target discovery, movement uncertainty
- **Scripts**: `eval-oepnai-multi-episode.py`, `eval-baichuan-multi-episode.py`

### 4. Myriad Echoes
- **Modalities**: Video + Audio + Image
- **Task**: Observe and reproduce audiovisual sequences
- **Evaluation**: Sequence parsing ability, coordinate prediction accuracy
- **Scripts**: `eval_baichuan_multi_episode.py`, `eval_openai_multi_episode.py`

### 5. Blasting Showdown
- **Modalities**: Image + Audio + Game State
- **Task**: Multi-AI model competitive battles
- **Features**: Real-time strategic decisions, multi-agent interaction
- **Scripts**: `start_ai_game.py`, `multi_model_game.py`

## 🚀 Quick Start

### 1. Environment Setup
```bash
pip install pygame numpy pillow requests gymnasium opencv-python moviepy
```

### 2. API Configuration
Set API information in configuration files within each game directory:
```python
API_BASE = "your_api_endpoint"
API_KEY = "your_api_key"
MODEL_CHAT = "your_model_name"
```

### 3. Run Evaluations
```bash
# Whispered Pathfinding
cd eval/game/Whispered_Pathfinding
python test_openai.py --difficulty medium --rounds 5

# The Alchemist's Melody
cd eval/game/The_Alchemist-s_Melody
python run_mm_agent.py

# Phantom Soldiers in the Fog
cd eval/game/Phantom_Soldiers_in_the_Fog
python eval-oepnai-multi-episode.py --difficulty medium --num_episodes 10

# Myriad Echoes
cd eval/game/Myriad_Echoes
python eval_openai_multi_episode.py

# Blasting Showdown
cd eval/game/Blasting_Showdown
python start_ai_game.py --config model_config.json
```

## 📊 Evaluation Metrics

- **Success Rate**: Task completion percentage
- **Efficiency**: Average completion steps/time
- **Accuracy**: Action execution precision
- **Adaptability**: Cross-difficulty performance consistency
- **Multimodal Understanding**: Effectiveness of utilizing information from each modality

## 🛠️ Supported Models

- **OpenAI**: All models compatible with OpenAI API interface
- **Baichuan**: Multimodal FastAPI service
- **General**: API endpoints compatible with OpenAI format

## 📁 Project Structure

```
omniplay/
├── eval/game/                    # Game evaluation environments
│   ├── Whispered_Pathfinding/    # Maze pathfinding
│   ├── The_Alchemist-s_Melody/   # Musical sequences
│   ├── Phantom_Soldiers_in_the_Fog/  # Tactical command
│   ├── Myriad_Echoes/            # Rhythm memory
│   └── Blasting_Showdown/        # Bomberman battles
└── game/assets-necessay/         # Shared game resources
```

## ⚠️ Important Notes

- **Single Instance**: Avoid running multiple game instances simultaneously
- **Resource Requirements**: Ensure sufficient memory and computational resources
- **API Limitations**: Be mindful of model call frequency and costs
- **Dependency Check**: Some games require additional audio/video processing libraries

## 📄 License

Open-source project for research and evaluation purposes. Please comply with relevant API service terms when using.

