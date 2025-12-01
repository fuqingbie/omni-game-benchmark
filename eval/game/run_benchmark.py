#!/usr/bin/env python3
"""
Unified Benchmark Runner for Omni-Game-Benchmark

This script provides a single entry point for running game evaluations
across multiple games and models with standardized configuration.

Usage:
    python run_benchmark.py --config benchmark_config.yaml
    python run_benchmark.py --game alchemist_melody --model gpt-4o --episodes 10
    python run_benchmark.py --list-games
    python run_benchmark.py --list-models
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import common modules
from game.common import (
    ModelRegistry,
    ModelCapability,
    get_default_registry,
    GameRegistry,
    get_game_registry,
    register_game,
    BenchmarkRunner,
    RunConfig,
    StatisticsAnalyzer,
    BenchmarkResult,
)
from game.common.game_registry import register_builtin_games


def setup_registries():
    """Initialize model and game registries with default entries."""
    # Register built-in games
    register_builtin_games()
    
    # Register common models
    model_registry = get_default_registry()
    
    # OpenAI models
    model_registry.register(
        name="gpt-4o",
        api_base="https://api.openai.com/v1",
        capability_preset="openai",
    )
    
    model_registry.register(
        name="gpt-4o-frames",
        api_base="https://api.openai.com/v1",
        capability_preset="openai-frames",
    )
    
    # Baichuan
    model_registry.register(
        name="baichuan",
        api_base="https://api.baichuan-ai.com/v1",
        capability_preset="baichuan",
    )
    
    # Qwen
    model_registry.register(
        name="qwen-vl",
        api_base="https://dashscope.aliyuncs.com/api/v1",
        capability_preset="qwen",
    )
    
    # Gemini
    model_registry.register(
        name="gemini-pro",
        api_base="https://generativelanguage.googleapis.com/v1beta",
        capability_preset="gemini",
    )


def list_games():
    """List all registered games."""
    registry = get_game_registry()
    games = registry.list_games()
    
    print("Registered Games:")
    print("-" * 60)
    for name in sorted(games):
        entry = registry.get(name)
        if entry:
            tags = ", ".join(entry.tags) if entry.tags else "No tags"
            print(f"  {name}")
            print(f"    Description: {entry.description or 'No description'}")
            print(f"    Tags: {tags}")
            print(f"    Default Episodes: {entry.default_episodes}")
            print()


def list_models():
    """List all registered models."""
    registry = get_default_registry()
    models = registry.list_models()
    
    print("Registered Models:")
    print("-" * 60)
    for name in sorted(models):
        config = registry.get(name)
        if config:
            cap = config.capability
            print(f"  {name}")
            print(f"    API Base: {config.api_base}")
            print(f"    Video: {cap.video.name}")
            print(f"    Audio: {cap.audio.name}")
            print(f"    Has API Key: {'Yes' if config.api_key else 'No (set via env)'}")
            print()


def run_single_evaluation(
    game: str,
    model: str,
    episodes: int = 10,
    difficulty: str = "normal",
    output_dir: str = "results",
    verbose: bool = False,
):
    """Run a single game-model evaluation."""
    print(f"\n{'='*60}")
    print(f"Running: {game} with {model}")
    print(f"Episodes: {episodes}, Difficulty: {difficulty}")
    print(f"{'='*60}\n")
    
    config = RunConfig(
        games=[game],
        models=[model],
        episodes_per_game=episodes,
        difficulty=difficulty,
        output_dir=output_dir,
        save_episodes=True,
    )
    
    runner = BenchmarkRunner(config)
    result = runner.run()
    
    # Print summary
    if result.game_results:
        gr = result.game_results[0]
        print(f"\n{'='*60}")
        print("Evaluation Complete")
        print(f"{'='*60}")
        print(f"Game: {gr.game_name}")
        print(f"Model: {gr.model_name}")
        print(f"Episodes: {gr.total_episodes}")
        print(f"Mean Score: {gr.mean_score:.2f} ± {gr.std_score:.2f}")
        print(f"Win Rate: {gr.win_rate:.1%}")
        print(f"Duration: {gr.total_duration_seconds:.1f}s")
    
    return result


def run_from_config(config_path: str):
    """Run benchmark from configuration file."""
    print(f"Loading configuration from: {config_path}")
    
    config = RunConfig.from_file(config_path)
    runner = BenchmarkRunner(config)
    result = runner.run()
    
    # Generate analysis report
    analyzer = StatisticsAnalyzer()
    analysis = analyzer.analyze_benchmark(result)
    
    # Save analysis
    output_dir = Path(config.output_dir)
    analyzer.generate_report(
        result,
        output_dir / "analysis_report.json",
        format="json"
    )
    analyzer.generate_report(
        result,
        output_dir / "analysis_report.md",
        format="md"
    )
    
    print(f"\nAnalysis reports saved to {output_dir}")
    
    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Omni-Game-Benchmark Unified Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available games and models
  python run_benchmark.py --list-games
  python run_benchmark.py --list-models
  
  # Run single evaluation
  python run_benchmark.py --game alchemist_melody --model gpt-4o --episodes 10
  
  # Run from config file
  python run_benchmark.py --config benchmark_config.yaml
  
  # Run multiple games
  python run_benchmark.py --games alchemist_melody phantom_soldiers --model gpt-4o
        """
    )
    
    # Info commands
    parser.add_argument("--list-games", action="store_true", help="List registered games")
    parser.add_argument("--list-models", action="store_true", help="List registered models")
    
    # Config file
    parser.add_argument("--config", "-c", type=str, help="Path to configuration file")
    
    # Single run options
    parser.add_argument("--game", type=str, help="Single game to evaluate")
    parser.add_argument("--games", nargs="+", help="Multiple games to evaluate")
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--models", nargs="+", help="Multiple models to use")
    
    # Evaluation parameters
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per game")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--difficulty", type=str, default="normal", 
                       choices=["easy", "normal", "hard"], help="Game difficulty")
    
    # Output options
    parser.add_argument("--output", "-o", type=str, default="results", help="Output directory")
    parser.add_argument("--save-video", action="store_true", help="Save video recordings")
    parser.add_argument("--save-steps", action="store_true", help="Save detailed step data")
    
    # Execution options
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Initialize registries
    setup_registries()
    
    # Handle info commands
    if args.list_games:
        list_games()
        return 0
    
    if args.list_models:
        list_models()
        return 0
    
    # Run from config file
    if args.config:
        run_from_config(args.config)
        return 0
    
    # Validate required arguments for direct run
    games = []
    if args.game:
        games.append(args.game)
    if args.games:
        games.extend(args.games)
    
    models = []
    if args.model:
        models.append(args.model)
    if args.models:
        models.extend(args.models)
    
    if not games:
        print("Error: No games specified. Use --game or --games, or --list-games to see available games.")
        return 1
    
    if not models:
        print("Error: No models specified. Use --model or --models, or --list-models to see available models.")
        return 1
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / f"run_{timestamp}"
    
    # Create run config
    config = RunConfig(
        games=games,
        models=models,
        episodes_per_game=args.episodes,
        max_steps_per_episode=args.max_steps,
        difficulty=args.difficulty,
        output_dir=str(output_dir),
        save_video=args.save_video,
        save_steps=args.save_steps,
        seed=args.seed,
    )
    
    # Run benchmark
    runner = BenchmarkRunner(config)
    result = runner.run()
    
    # Generate reports
    analyzer = StatisticsAnalyzer()
    analyzer.generate_report(result, output_dir / "report.md", format="md")
    analyzer.generate_report(result, output_dir / "report.json", format="json")
    
    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"Games evaluated: {result.games_evaluated}")
    print(f"Models evaluated: {result.models_evaluated}")
    print(f"Total episodes: {result.total_episodes}")
    
    # Print top results
    leaderboard = result.get_leaderboard("mean_score")
    if leaderboard:
        print("\nTop Results:")
        for i, entry in enumerate(leaderboard[:5], 1):
            print(f"  {i}. {entry['game']}/{entry['model']}: {entry['mean_score']:.2f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
