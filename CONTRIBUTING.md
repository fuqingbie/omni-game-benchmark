# Contributing to OmniPlay

Thank you for your interest in contributing to OmniPlay! This document provides guidelines and instructions for contributing.

## 🤝 Ways to Contribute

### 1. 🎮 Add a New Game

To add a new game to the benchmark:

1. **Create game directory** under `eval/game/YourGameName/`

2. **Implement the game environment**:
   ```python
   from eval.game.common import BaseGameEnv, GameInfo
   
   class YourGameEnv(BaseGameEnv):
       def __init__(self, config=None):
           super().__init__(config)
           # Initialize your game
       
       def reset(self):
           # Reset game state
           return observation
       
       def step(self, action):
           # Execute action
           return observation, reward, done, info
       
       def get_game_info(self) -> GameInfo:
           return GameInfo(
               name="your_game",
               description="Your game description",
               modalities=["image", "audio"],
               difficulty_levels=["easy", "medium", "hard"]
           )
   ```

3. **Register your game** in `eval/game/common/game_registry.py`:
   ```python
   register_game(
       name="your_game",
       env_class=YourGameEnv,
       description="Your game description",
       tags=["puzzle", "audio"]
   )
   ```

4. **Add evaluation scripts**:
   - `eval_openai.py` - OpenAI model evaluation
   - `eval_baichuan.py` - Baichuan model evaluation (if applicable)

5. **Create README.md** with:
   - Game description and rules
   - Evaluation metrics
   - Usage instructions

### 2. 🤖 Add a New Model

To add support for a new model:

1. **Register the model** in your evaluation script:
   ```python
   from eval.game.common import get_default_registry
   
   registry = get_default_registry()
   registry.register(
       name="your-model",
       api_base="https://api.yourprovider.com/v1",
       capability_preset="openai",  # or create custom capability
   )
   ```

2. **For custom capabilities**, add a new preset in `model_registry.py`:
   ```python
   CAPABILITY_PRESETS["your-provider"] = ModelCapability(
       video=VideoCapability.FRAME_EXTRACT,
       audio=AudioCapability.DIRECT_AUDIO,
       image=ImageCapability.BASE64_IMAGE,
       frame_interval=0.5,
       max_frames=10
   )
   ```

3. **Test your model** on all games and submit results

### 3. 📊 Submit Evaluation Results

To add your results to the leaderboard:

1. **Run the unified benchmark**:
   ```bash
   python eval/game/run_benchmark.py \
       --model your-model \
       --episodes 50 \
       --output-dir ./results/your-model
   ```

2. **Verify results format**:
   ```json
   {
     "game": "alchemist_melody",
     "model": "your-model",
     "episodes": [...],
     "metrics": {
       "accuracy": 0.85,
       "avg_steps": 15.2
     }
   }
   ```

3. **Open a Pull Request** updating `docs/LEADERBOARD.md`

### 4. 🐛 Report Issues

When reporting issues, please include:

- **Environment**: Python version, OS, dependencies
- **Game**: Which game(s) affected
- **Model**: Which model(s) tested
- **Steps to reproduce**: Minimal code/commands
- **Error messages**: Full traceback if applicable
- **Expected vs actual behavior**

### 5. 📖 Improve Documentation

Documentation improvements are always welcome:

- Fix typos and clarify instructions
- Add usage examples
- Improve API documentation
- Translate to other languages

---

## 🔧 Development Setup

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

```bash
# Clone the repository
git clone https://github.com/fuqingbie/omni-game-benchmark.git
cd omni-game-benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy

# Run tests
pytest tests/
```

### Code Style

We use the following tools for code quality:

- **Black** for formatting: `black .`
- **Flake8** for linting: `flake8 .`
- **MyPy** for type checking: `mypy .`

Please run these before submitting PRs:

```bash
black eval/
flake8 eval/
mypy eval/
```

---

## 📋 Pull Request Guidelines

### Before Submitting

- [ ] Code follows the project style guide
- [ ] Tests pass locally
- [ ] Documentation updated if needed
- [ ] Commit messages are clear and descriptive

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
How was this tested?

## Related Issues
Fixes #123
```

### Review Process

1. Submit PR against `main` branch
2. Automated checks run (linting, tests)
3. Maintainer reviews code
4. Address feedback if any
5. PR merged after approval

---

## 📜 Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Accept responsibility for mistakes

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Publishing others' private information
- Other unprofessional conduct

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## 🙏 Acknowledgments

Thank you to all contributors who help make OmniPlay better!

<p align="center">
  <a href="https://github.com/fuqingbie/omni-game-benchmark/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=fuqingbie/omni-game-benchmark" />
  </a>
</p>

---

Questions? Open an issue or reach out to the maintainers!
