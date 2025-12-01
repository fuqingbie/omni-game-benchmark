"""
Statistics - Unified statistics analysis module.

This module provides tools for analyzing benchmark results,
computing metrics, and generating reports.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .result_schema import (
    EpisodeResult,
    GameResult,
    BenchmarkResult,
)


@dataclass
class MetricScore:
    """A score for a single metric."""
    name: str
    value: float
    max_value: float = 100.0
    min_value: float = 0.0
    weight: float = 1.0
    
    @property
    def normalized(self) -> float:
        """Get normalized score (0-100)."""
        if self.max_value == self.min_value:
            return 50.0
        return ((self.value - self.min_value) / (self.max_value - self.min_value)) * 100
    
    def to_grade(self) -> str:
        """Convert score to letter grade."""
        normalized = self.normalized
        if normalized >= 90:
            return "A"
        elif normalized >= 80:
            return "B"
        elif normalized >= 70:
            return "C"
        elif normalized >= 60:
            return "D"
        else:
            return "F"


@dataclass
class GameMetrics:
    """Metrics for a game evaluation."""
    game_name: str
    model_name: str
    
    # Basic metrics
    win_rate: float = 0.0
    mean_score: float = 0.0
    std_score: float = 0.0
    mean_reward: float = 0.0
    
    # Efficiency metrics
    mean_steps: float = 0.0
    score_per_step: float = 0.0
    
    # Consistency metrics
    coefficient_of_variation: float = 0.0  # std/mean
    
    # Detailed scores by dimension
    dimension_scores: Dict[str, MetricScore] = field(default_factory=dict)
    
    # Overall composite score
    overall_score: float = 0.0
    overall_grade: str = "F"
    
    def compute_overall_score(self):
        """Compute overall score from dimension scores."""
        if not self.dimension_scores:
            self.overall_score = 0.0
            self.overall_grade = "F"
            return
        
        total_weight = sum(s.weight for s in self.dimension_scores.values())
        if total_weight == 0:
            self.overall_score = 0.0
        else:
            weighted_sum = sum(
                s.normalized * s.weight for s in self.dimension_scores.values()
            )
            self.overall_score = weighted_sum / total_weight
        
        # Determine grade
        if self.overall_score >= 90:
            self.overall_grade = "A"
        elif self.overall_score >= 80:
            self.overall_grade = "B"
        elif self.overall_score >= 70:
            self.overall_grade = "C"
        elif self.overall_score >= 60:
            self.overall_grade = "D"
        else:
            self.overall_grade = "F"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "game_name": self.game_name,
            "model_name": self.model_name,
            "win_rate": self.win_rate,
            "mean_score": self.mean_score,
            "std_score": self.std_score,
            "mean_reward": self.mean_reward,
            "mean_steps": self.mean_steps,
            "score_per_step": self.score_per_step,
            "coefficient_of_variation": self.coefficient_of_variation,
            "dimension_scores": {
                name: {
                    "value": score.value,
                    "normalized": score.normalized,
                    "grade": score.to_grade(),
                }
                for name, score in self.dimension_scores.items()
            },
            "overall_score": self.overall_score,
            "overall_grade": self.overall_grade,
        }


class StatisticsAnalyzer:
    """
    Analyzer for computing statistics from benchmark results.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        self._game_specific_scorers: Dict[str, callable] = {}
        
        # Default dimension weights
        self.dimension_weights = {
            "accuracy": 2.0,      # Correctness of actions
            "efficiency": 1.5,   # Score per step
            "consistency": 1.0,  # Low variance
            "speed": 0.5,        # Steps to completion
            "win_rate": 2.0,     # Win percentage
            "reasoning": 1.5,    # For games with reasoning requirements
        }
    
    def analyze_game_result(self, game_result: GameResult) -> GameMetrics:
        """
        Analyze a game result and compute metrics.
        
        Args:
            game_result: The game result to analyze
            
        Returns:
            GameMetrics with computed metrics
        """
        metrics = GameMetrics(
            game_name=game_result.game_name,
            model_name=game_result.model_name,
        )
        
        if not game_result.episodes:
            return metrics
        
        # Basic metrics (already computed in GameResult)
        metrics.win_rate = game_result.win_rate
        metrics.mean_score = game_result.mean_score
        metrics.std_score = game_result.std_score
        metrics.mean_reward = game_result.mean_reward
        
        # Compute efficiency metrics
        total_steps = sum(e.total_steps for e in game_result.episodes)
        metrics.mean_steps = total_steps / len(game_result.episodes)
        
        if metrics.mean_steps > 0:
            metrics.score_per_step = metrics.mean_score / metrics.mean_steps
        
        # Coefficient of variation
        if metrics.mean_score != 0:
            metrics.coefficient_of_variation = metrics.std_score / abs(metrics.mean_score)
        
        # Compute dimension scores
        metrics.dimension_scores = self._compute_dimension_scores(game_result, metrics)
        
        # Apply game-specific scoring if available
        if game_result.game_name in self._game_specific_scorers:
            custom_scores = self._game_specific_scorers[game_result.game_name](game_result)
            metrics.dimension_scores.update(custom_scores)
        
        # Compute overall score
        metrics.compute_overall_score()
        
        return metrics
    
    def _compute_dimension_scores(
        self,
        game_result: GameResult,
        metrics: GameMetrics
    ) -> Dict[str, MetricScore]:
        """Compute scores for each dimension."""
        scores = {}
        
        # Win rate score
        scores["win_rate"] = MetricScore(
            name="Win Rate",
            value=metrics.win_rate * 100,
            max_value=100.0,
            min_value=0.0,
            weight=self.dimension_weights.get("win_rate", 2.0),
        )
        
        # Efficiency score (score per step, normalized)
        # Higher is better, normalize against reasonable max
        max_efficiency = 1.0  # Adjust based on game
        scores["efficiency"] = MetricScore(
            name="Efficiency",
            value=min(metrics.score_per_step, max_efficiency) * 100,
            max_value=100.0,
            min_value=0.0,
            weight=self.dimension_weights.get("efficiency", 1.5),
        )
        
        # Consistency score (based on coefficient of variation)
        # Lower CV is better, so invert
        consistency = max(0, 100 - metrics.coefficient_of_variation * 100)
        scores["consistency"] = MetricScore(
            name="Consistency",
            value=consistency,
            max_value=100.0,
            min_value=0.0,
            weight=self.dimension_weights.get("consistency", 1.0),
        )
        
        return scores
    
    def analyze_benchmark(self, benchmark_result: BenchmarkResult) -> Dict[str, Any]:
        """
        Analyze complete benchmark results.
        
        Args:
            benchmark_result: The benchmark result to analyze
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "benchmark_name": benchmark_result.benchmark_name,
            "summary": {},
            "by_game": {},
            "by_model": {},
            "leaderboard": [],
        }
        
        # Analyze each game result
        game_metrics: List[GameMetrics] = []
        for gr in benchmark_result.game_results:
            metrics = self.analyze_game_result(gr)
            game_metrics.append(metrics)
            
            # Store by game
            if gr.game_name not in analysis["by_game"]:
                analysis["by_game"][gr.game_name] = []
            analysis["by_game"][gr.game_name].append(metrics.to_dict())
            
            # Store by model
            if gr.model_name not in analysis["by_model"]:
                analysis["by_model"][gr.model_name] = []
            analysis["by_model"][gr.model_name].append(metrics.to_dict())
        
        # Compute summary statistics
        if game_metrics:
            analysis["summary"] = {
                "total_games": len(set(m.game_name for m in game_metrics)),
                "total_models": len(set(m.model_name for m in game_metrics)),
                "total_evaluations": len(game_metrics),
                "overall_mean_score": np.mean([m.overall_score for m in game_metrics]),
                "overall_mean_win_rate": np.mean([m.win_rate for m in game_metrics]),
            }
        
        # Build leaderboard
        analysis["leaderboard"] = sorted(
            [m.to_dict() for m in game_metrics],
            key=lambda x: x["overall_score"],
            reverse=True
        )
        
        return analysis
    
    def compare_models(
        self,
        benchmark_result: BenchmarkResult,
        models: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare performance across models.
        
        Args:
            benchmark_result: Benchmark results
            models: Models to compare (all if None)
            
        Returns:
            Comparison results
        """
        comparison = {
            "models": [],
            "by_game": {},
        }
        
        # Filter results
        results_to_compare = benchmark_result.game_results
        if models:
            results_to_compare = [
                gr for gr in results_to_compare if gr.model_name in models
            ]
        
        # Group by model
        model_results: Dict[str, List[GameResult]] = {}
        for gr in results_to_compare:
            if gr.model_name not in model_results:
                model_results[gr.model_name] = []
            model_results[gr.model_name].append(gr)
        
        # Compute aggregate metrics per model
        for model_name, results in model_results.items():
            metrics_list = [self.analyze_game_result(gr) for gr in results]
            
            model_summary = {
                "model": model_name,
                "games_played": len(results),
                "mean_overall_score": np.mean([m.overall_score for m in metrics_list]),
                "mean_win_rate": np.mean([m.win_rate for m in metrics_list]),
                "total_episodes": sum(gr.total_episodes for gr in results),
            }
            comparison["models"].append(model_summary)
        
        # Sort by overall score
        comparison["models"].sort(
            key=lambda x: x["mean_overall_score"],
            reverse=True
        )
        
        return comparison
    
    def register_game_scorer(self, game_name: str, scorer: callable):
        """
        Register a custom scorer for a specific game.
        
        Args:
            game_name: Name of the game
            scorer: Function that takes GameResult and returns Dict[str, MetricScore]
        """
        self._game_specific_scorers[game_name] = scorer
    
    def generate_report(
        self,
        benchmark_result: BenchmarkResult,
        output_path: str,
        format: str = "json"
    ):
        """
        Generate an analysis report.
        
        Args:
            benchmark_result: Results to analyze
            output_path: Path for output file
            format: Output format (json, html, md)
        """
        analysis = self.analyze_benchmark(benchmark_result)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        elif format == "md" or format == "markdown":
            self._generate_markdown_report(analysis, output_path)
        
        elif format == "html":
            self._generate_html_report(analysis, output_path)
    
    def _generate_markdown_report(self, analysis: Dict[str, Any], output_path: Path):
        """Generate markdown report."""
        lines = [
            "# Benchmark Analysis Report",
            "",
            f"Generated: {analysis['timestamp']}",
            f"Benchmark: {analysis['benchmark_name']}",
            "",
            "## Summary",
            "",
            f"- Total Games: {analysis['summary'].get('total_games', 0)}",
            f"- Total Models: {analysis['summary'].get('total_models', 0)}",
            f"- Mean Score: {analysis['summary'].get('overall_mean_score', 0):.1f}",
            f"- Mean Win Rate: {analysis['summary'].get('overall_mean_win_rate', 0):.1%}",
            "",
            "## Leaderboard",
            "",
            "| Rank | Game | Model | Score | Grade | Win Rate |",
            "|------|------|-------|-------|-------|----------|",
        ]
        
        for i, entry in enumerate(analysis["leaderboard"][:20], 1):
            lines.append(
                f"| {i} | {entry['game_name']} | {entry['model_name']} | "
                f"{entry['overall_score']:.1f} | {entry['overall_grade']} | "
                f"{entry['win_rate']:.1%} |"
            )
        
        lines.extend([
            "",
            "## Performance by Model",
            "",
        ])
        
        for model_name, results in analysis["by_model"].items():
            mean_score = np.mean([r["overall_score"] for r in results])
            lines.append(f"### {model_name}")
            lines.append(f"- Average Score: {mean_score:.1f}")
            lines.append("")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    def _generate_html_report(self, analysis: Dict[str, Any], output_path: Path):
        """Generate HTML report."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Report - {analysis['benchmark_name']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .grade-A {{ color: green; font-weight: bold; }}
        .grade-B {{ color: blue; }}
        .grade-C {{ color: orange; }}
        .grade-D {{ color: red; }}
        .grade-F {{ color: darkred; }}
    </style>
</head>
<body>
    <h1>Benchmark Analysis Report</h1>
    <p>Generated: {analysis['timestamp']}</p>
    
    <h2>Summary</h2>
    <ul>
        <li>Total Games: {analysis['summary'].get('total_games', 0)}</li>
        <li>Total Models: {analysis['summary'].get('total_models', 0)}</li>
        <li>Mean Score: {analysis['summary'].get('overall_mean_score', 0):.1f}</li>
    </ul>
    
    <h2>Leaderboard</h2>
    <table>
        <tr>
            <th>Rank</th>
            <th>Game</th>
            <th>Model</th>
            <th>Score</th>
            <th>Grade</th>
            <th>Win Rate</th>
        </tr>
"""
        
        for i, entry in enumerate(analysis["leaderboard"][:20], 1):
            grade = entry['overall_grade']
            html += f"""        <tr>
            <td>{i}</td>
            <td>{entry['game_name']}</td>
            <td>{entry['model_name']}</td>
            <td>{entry['overall_score']:.1f}</td>
            <td class="grade-{grade}">{grade}</td>
            <td>{entry['win_rate']:.1%}</td>
        </tr>
"""
        
        html += """    </table>
</body>
</html>"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)


def compute_game_metrics(game_result: GameResult) -> GameMetrics:
    """
    Convenience function to compute metrics for a game result.
    
    Args:
        game_result: The game result to analyze
        
    Returns:
        GameMetrics
    """
    analyzer = StatisticsAnalyzer()
    return analyzer.analyze_game_result(game_result)


# Game-specific scorers
def alchemist_melody_scorer(game_result: GameResult) -> Dict[str, MetricScore]:
    """Custom scorer for Alchemist's Melody game."""
    scores = {}
    
    # Audio understanding dimension
    # Based on sequence completion rate
    if game_result.episodes:
        sequence_completions = []
        for ep in game_result.episodes:
            if ep.game_metrics:
                seq_progress = ep.game_metrics.get("sequence_progress", 0)
                seq_length = ep.game_metrics.get("sequence_length", 1)
                sequence_completions.append(seq_progress / seq_length)
        
        if sequence_completions:
            audio_score = np.mean(sequence_completions) * 100
            scores["audio_understanding"] = MetricScore(
                name="Audio Understanding",
                value=audio_score,
                max_value=100.0,
                min_value=0.0,
                weight=2.0,
            )
    
    return scores


def maze_scorer(game_result: GameResult) -> Dict[str, MetricScore]:
    """Custom scorer for Whispered Pathfinding (Maze) game."""
    scores = {}
    
    # Navigation efficiency
    if game_result.episodes:
        efficiencies = []
        for ep in game_result.episodes:
            if ep.game_metrics:
                optimal_steps = ep.game_metrics.get("optimal_steps", ep.total_steps)
                if optimal_steps > 0:
                    efficiency = optimal_steps / ep.total_steps
                    efficiencies.append(efficiency)
        
        if efficiencies:
            nav_score = np.mean(efficiencies) * 100
            scores["navigation"] = MetricScore(
                name="Navigation Efficiency",
                value=nav_score,
                max_value=100.0,
                min_value=0.0,
                weight=1.5,
            )
    
    return scores


# Register default game scorers
def register_default_scorers(analyzer: StatisticsAnalyzer):
    """Register default game-specific scorers."""
    analyzer.register_game_scorer("alchemist_melody", alchemist_melody_scorer)
    analyzer.register_game_scorer("whispered_pathfinding", maze_scorer)
