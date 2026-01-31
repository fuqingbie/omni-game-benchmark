# 🏆 OmniPlay Leaderboard

> **Last Updated**: October 2025

This document contains the complete leaderboard for the OmniPlay benchmark, including detailed per-game metrics and historical performance tracking.

## 📊 Overall Ranking

| Rank | Model | Provider | Whispered Pathfinding | Myriad Echoes | Phantom Soldiers | Alchemist's Melody | Blasting Showdown |
|:----:|:------|:---------|:---------------------:|:-------------:|:----------------:|:------------------:|:-----------------:|
| Ref | **Human Expert** | - | 100.0 | 100.0 | 100.0 | 100.0 | - |
| 🥇 | **Gemini 2.5 Pro** | Google | 97.5 | 223.4 | 83.2 | 28.4 | 36.11% |
| 🥈 | **Gemini 2.5 Flash** | Google | 95.5 | 23.7 | 49.1 | 10.5 | 28.95% |
| 🥉 | **Baichuan-Omni-1.5** | Baichuan | 88.7 | -2.3 | -3.5 | 10.2 | 17.65% |
| 4 | MiniCPM-o-2.6 | OpenBMB | 86.4 | -1.3 | -30.7 | 7.7 | 19.35% |
| 5 | VITA-1.5 | Open | 81.9 | -3.6 | -52.2 | -8.9 | 7.41% |
| 6 | Qwen-2.5-Omni | Alibaba | 73.6 | -2.7 | -7.8 | 9.2 | 11.76% |

> 📝 **Note**: 
> - Scores for games 1-4 are normalized relative to Human Expert (100.0 = human-level performance)
> - Negative scores indicate performance below random baseline
> - Blasting Showdown uses Win Rate (%) as the metric
> - "-" indicates not evaluated or not applicable

---

## 🎮 Per-Game Results

### 🔊 1. Whispered Pathfinding (Navigation)

**Task**: Navigate maze using voice guidance  
**Modalities**: Image + Audio + Vector State  
**Key Skills**: Audio comprehension, spatial navigation

| Rank | Model | Score |
|:----:|:------|:-----:|
| Ref | Human Expert | **100.0** |
| 1 | Gemini 2.5 Pro | **97.5** |
| 2 | Gemini 2.5 Flash | **95.5** |
| 3 | Baichuan-Omni-1.5 | **88.7** |
| 4 | MiniCPM-o-2.6 | **86.4** |
| 5 | VITA-1.5 | **81.9** |
| 6 | Qwen-2.5-Omni | **73.6** |

**Insights**:
- All models achieve reasonable performance on this task
- Gemini 2.5 Pro nearly matches human-level performance (97.5%)
- Audio comprehension is generally strong across models

---

### 🎵 2. Myriad Echoes (Memory)

**Task**: Observe and reproduce audiovisual sequences  
**Modalities**: Video + Audio + Image  
**Key Skills**: Sequence memory, coordinate prediction

| Rank | Model | Score |
|:----:|:------|:-----:|
| Ref | Human Expert | **100.0** |
| 1 | Gemini 2.5 Pro | **223.4** |
| 2 | Gemini 2.5 Flash | **23.7** |
| 3 | MiniCPM-o-2.6 | **-1.3** |
| 4 | Baichuan-Omni-1.5 | **-2.3** |
| 5 | Qwen-2.5-Omni | **-2.7** |
| 6 | VITA-1.5 | **-3.6** |

**Insights**:
- Gemini 2.5 Pro significantly outperforms human baseline (223.4%)
- Most models struggle with this memory-intensive task
- Large gap between Gemini models and other competitors

---

### 👻 3. Phantom Soldiers in the Fog (Strategy)

**Task**: Command units to discover hidden targets  
**Modalities**: Video + Audio + Vector State  
**Key Skills**: Tactical reasoning, spatial awareness, uncertainty handling

| Rank | Model | Score |
|:----:|:------|:-----:|
| Ref | Human Expert | **100.0** |
| 1 | Gemini 2.5 Pro | **83.2** |
| 2 | Gemini 2.5 Flash | **49.1** |
| 3 | Baichuan-Omni-1.5 | **-3.5** |
| 4 | Qwen-2.5-Omni | **-7.8** |
| 5 | MiniCPM-o-2.6 | **-30.7** |
| 6 | VITA-1.5 | **-52.2** |

**Insights**:
- This is the most challenging game for most models
- Only Gemini models achieve positive scores
- Handling fog-of-war uncertainty is the key differentiator

---

### 🧪 4. The Alchemist's Melody (Reasoning)

**Task**: Learn color-note mappings and reproduce sequences  
**Modalities**: Image + Audio + State  
**Key Skills**: Audio-visual association, pattern learning

| Rank | Model | Score |
|:----:|:------|:-----:|
| Ref | Human Expert | **100.0** |
| 1 | Gemini 2.5 Pro | **28.4** |
| 2 | Gemini 2.5 Flash | **10.5** |
| 3 | Baichuan-Omni-1.5 | **10.2** |
| 4 | Qwen-2.5-Omni | **9.2** |
| 5 | MiniCPM-o-2.6 | **7.7** |
| 6 | VITA-1.5 | **-8.9** |

**Insights**:
- All models significantly underperform compared to humans
- Audio-visual reasoning remains challenging for current LMMs
- Gemini 2.5 Pro leads but still achieves only ~28% of human performance

---

### 💣 5. Blasting Showdown (Combat)

**Task**: Multi-agent Bomberman battles  
**Modalities**: Image + Audio + Game State  
**Key Skills**: Strategic planning, real-time decision making

| Rank | Model | Win Rate |
|:----:|:------|:--------:|
| 1 | Gemini 2.5 Pro | **36.11%** |
| 2 | Gemini 2.5 Flash | **28.95%** |
| 3 | MiniCPM-o-2.6 | **19.35%** |
| 4 | Baichuan-Omni-1.5 | **17.65%** |
| 5 | Qwen-2.5-Omni | **11.76%** |
| 6 | VITA-1.5 | **7.41%** |

**Insights**:
- No human baseline available for this competitive game
- Gemini 2.5 Pro achieves highest win rate at 36.11%
- Real-time strategic decision-making remains challenging

---

## 📈 Key Findings

### Model Ranking Summary

1. **Gemini 2.5 Pro** - Best overall performance across all games
2. **Gemini 2.5 Flash** - Strong second place, good balance of performance
3. **Baichuan-Omni-1.5** - Best among Chinese models
4. **MiniCPM-o-2.6** - Competitive in navigation and combat
5. **VITA-1.5** - Good at navigation, struggles with strategy
6. **Qwen-2.5-Omni** - Moderate performance across tasks

### Task Difficulty (for AI models)

| Difficulty | Game | Observation |
|:-----------|:-----|:------------|
| 🟢 Easy | Whispered Pathfinding | Most models achieve >70% of human performance |
| 🟡 Medium | Myriad Echoes | Only Gemini models perform well |
| 🔴 Hard | Phantom Soldiers | Most models score negative |
| 🔴 Hard | Alchemist's Melody | Best model at ~28% of human level |
| 🟡 Medium | Blasting Showdown | Competitive, but low win rates overall |

---

## 📈 Methodology

### Score Calculation

- **Games 1-4**: Scores normalized relative to Human Expert baseline (100.0)
- **Game 5**: Win Rate percentage in multi-agent battles
- **Negative scores**: Indicate performance below random baseline

### Evaluation Protocol

- **Episodes per game**: 50 (10 per difficulty level where applicable)
- **Random seeds**: Fixed for reproducibility
- **API settings**: Temperature=0.0, deterministic mode
- **Timeout**: 5 minutes per episode

---

## 📥 Submit Your Results

Want to add your model to the leaderboard? 

1. Run the unified benchmark:
   ```bash
   python eval/game/run_benchmark.py --model your-model --episodes 50
   ```

2. Open a Pull Request with your results

### Submission Requirements

- [ ] Minimum 50 episodes per game
- [ ] All 5 games evaluated
- [ ] Results in standard JSON format
- [ ] Model configuration documented
- [ ] Reproducible evaluation settings

---

<p align="center">
  <a href="../README.md">← Back to README</a>
</p>
