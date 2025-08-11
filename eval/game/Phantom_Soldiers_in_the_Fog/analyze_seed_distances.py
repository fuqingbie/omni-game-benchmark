#!/usr/bin/env python3
"""
Analyze distance between objectives and nearest team member for 30 seeds
This helps determine appropriate movement error coefficients and tolerance values.
"""

import json
import numpy as np
from typing import List, Dict, Tuple
from env import CoopCommandEnv, GameConfig, GameDifficulty
import matplotlib.pyplot as plt

def calculate_distance(pos1: Dict, pos2: Dict) -> float:
    """Calculate Euclidean distance between two positions."""
    return np.sqrt((pos1['x'] - pos2['x'])**2 + (pos1['y'] - pos2['y'])**2)

def analyze_seed(seed_index: int, difficulty: str = "medium") -> Dict:
    """Analyze a single seed for objective-member distances."""
    
    # Convert difficulty string to enum
    difficulty_map = {
        "normal": GameDifficulty.NORMAL,
        "medium": GameDifficulty.MEDIUM, 
        "hard": GameDifficulty.HARD
    }
    
    config = GameConfig(
        difficulty=difficulty_map[difficulty],
        seed_index=seed_index,
        enable_audio=False
    )
    
    env = CoopCommandEnv(config=config)
    state = env.start_game()
    
    results = {
        "seed": seed_index,
        "objectives": [],
        "team_positions": [],
        "distances": []
    }
    
    # Get team member positions
    team_members = state['team_status']['members']
    team_positions = []
    for member_id, member_info in team_members.items():
        team_positions.append({
            'name': member_info['name'],
            'role': member_info['role'],
            'position': member_info['position']
        })
    
    results["team_positions"] = team_positions
    
    # Get all objectives (including hidden ones)
    all_objectives = []
    
    # Visible objectives
    for obj in state['mission_objectives']:
        if 'target_position' in obj:
            all_objectives.append(obj)
    
    # Access hidden objectives directly from game logic
    for hidden_obj in env.game_logic.hidden_objectives:
        # Hidden objectives are dictionaries with target_position
        if isinstance(hidden_obj, dict) and 'target_position' in hidden_obj:
            all_objectives.append(hidden_obj)
    
    # Calculate distances for each objective
    for obj in all_objectives:
        obj_pos = obj['target_position']
        
        # Find distance to nearest team member
        min_distance = float('inf')
        nearest_member = None
        
        for member in team_positions:
            distance = calculate_distance(obj_pos, member['position'])
            if distance < min_distance:
                min_distance = distance
                nearest_member = member['name']
        
        obj_result = {
            'description': obj['description'],
            'position': obj_pos,
            'nearest_member': nearest_member,
            'distance_to_nearest': min_distance
        }
        
        results["objectives"].append(obj_result)
        results["distances"].append(min_distance)
    
    env.shutdown()
    return results

def analyze_all_seeds(num_seeds: int = 30, difficulty: str = "medium") -> Dict:
    """Analyze all seeds and compile statistics."""
    
    print(f"Analyzing {num_seeds} seeds with {difficulty} difficulty...")
    
    all_results = []
    all_distances = []
    
    for seed in range(num_seeds):
        print(f"Processing seed {seed}...")
        try:
            result = analyze_seed(seed, difficulty)
            all_results.append(result)
            all_distances.extend(result["distances"])
        except Exception as e:
            print(f"Error processing seed {seed}: {e}")
            continue
    
    # Calculate statistics
    if not all_distances:
        print("Warning: No objective distances found. Returning empty stats.")
        return {
            "num_seeds": len(all_results),
            "total_objectives": 0,
            "distance_stats": {},
            "recommendations": {},
            "seed_results": all_results
        }
    
    distances_array = np.array(all_distances)
    
    stats = {
        "num_seeds": len(all_results),
        "total_objectives": len(all_distances),
        "distance_stats": {
            "mean": float(np.mean(distances_array)),
            "median": float(np.median(distances_array)),
            "std": float(np.std(distances_array)),
            "min": float(np.min(distances_array)),
            "max": float(np.max(distances_array)),
            "percentiles": {
                "25th": float(np.percentile(distances_array, 25)),
                "75th": float(np.percentile(distances_array, 75)),
                "90th": float(np.percentile(distances_array, 90)),
                "95th": float(np.percentile(distances_array, 95))
            }
        },
        "recommendations": {},
        "seed_results": all_results
    }
    
    # Generate recommendations only if we have data
    if stats["distance_stats"]:
        mean_dist = stats["distance_stats"]["mean"]
        
        # Recommend movement error coefficients based on different roles
        # Current precision values: scout=0.5, medic=1.0, heavy=2.0, engineer=1.5
        stats["recommendations"] = {
            "movement_error_analysis": {
                "current_system": "distance * role_precision * 0.1",
                "average_movement_distance": mean_dist,
                "scout_avg_error": mean_dist * 0.5 * 0.1,
                "medic_avg_error": mean_dist * 1.0 * 0.1,
                "heavy_avg_error": mean_dist * 2.0 * 0.1,
                "engineer_avg_error": mean_dist * 1.5 * 0.1
            },
            "tolerance_analysis": {
                "current_tolerance": 5.0,
                "recommended_tolerance": max(5.0, mean_dist * 0.15),  # 15% of average distance
                "rationale": f"With average distance {mean_dist:.1f}, tolerance should be ~{mean_dist * 0.15:.1f}"
            },
            "coefficient_suggestions": {
                "conservative": 0.05,  # 5% error
                "moderate": 0.1,       # 10% error (current)
                "realistic": 0.15      # 15% error
            }
        }
    
    return stats

def create_visualizations(stats: Dict, save_plots: bool = True):
    """Create visualizations of the distance analysis."""
    
    all_distances = []
    for result in stats["seed_results"]:
        all_distances.extend(result["distances"])
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Histogram of distances
    ax1.hist(all_distances, bins=20, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Distance to Nearest Member')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Objective-Member Distances')
    ax1.axvline(stats["distance_stats"]["mean"], color='red', linestyle='--', 
                label=f'Mean: {stats["distance_stats"]["mean"]:.1f}')
    ax1.legend()
    
    # 2. Box plot
    ax2.boxplot(all_distances)
    ax2.set_ylabel('Distance')
    ax2.set_title('Distance Distribution Box Plot')
    ax2.set_xticklabels(['All Objectives'])
    
    # 3. Distance by seed
    seed_avg_distances = []
    for result in stats["seed_results"]:
        if result["distances"]:
            seed_avg_distances.append(np.mean(result["distances"]))
    
    ax3.plot(range(len(seed_avg_distances)), seed_avg_distances, 'o-')
    ax3.set_xlabel('Seed Index')
    ax3.set_ylabel('Average Distance')
    ax3.set_title('Average Distance per Seed')
    ax3.grid(True, alpha=0.3)
    
    # 4. Movement error predictions
    distances_range = np.linspace(0, max(all_distances), 100)
    coefficients = [0.05, 0.1, 0.15]
    roles = {'scout': 0.5, 'medic': 1.0, 'heavy': 2.0, 'engineer': 1.5}
    
    for coef in coefficients:
        for role_name, role_precision in roles.items():
            if role_name == 'scout':  # Only show scout for clarity
                errors = distances_range * role_precision * coef
                ax4.plot(distances_range, errors, label=f'{role_name} (coef={coef})')
    
    ax4.set_xlabel('Movement Distance')
    ax4.set_ylabel('Movement Error')
    ax4.set_title('Predicted Movement Error vs Distance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('seed_distance_analysis.png', dpi=300, bbox_inches='tight')
        print("Plots saved as 'seed_distance_analysis.png'")
    
    plt.show()

def print_summary(stats: Dict):
    """Print a summary of the analysis results."""
    
    print("\n" + "="*60)
    print(" SEED DISTANCE ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"Seeds analyzed: {stats['num_seeds']}")
    print(f"Total objectives: {stats['total_objectives']}")
    
    print(f"\nDistance Statistics:")
    dist_stats = stats["distance_stats"]
    print(f"  Mean distance: {dist_stats['mean']:.2f} units")
    print(f"  Median distance: {dist_stats['median']:.2f} units")
    print(f"  Standard deviation: {dist_stats['std']:.2f} units")
    print(f"  Range: {dist_stats['min']:.2f} - {dist_stats['max']:.2f} units")
    print(f"  25th percentile: {dist_stats['percentiles']['25th']:.2f} units")
    print(f"  75th percentile: {dist_stats['percentiles']['75th']:.2f} units")
    print(f"  90th percentile: {dist_stats['percentiles']['90th']:.2f} units")
    
    print(f"\nMovement Error Analysis (current system):")
    error_analysis = stats["recommendations"]["movement_error_analysis"]
    print(f"  Average movement distance: {error_analysis['average_movement_distance']:.2f}")
    print(f"  Scout average error: {error_analysis['scout_avg_error']:.2f}")
    print(f"  Medic average error: {error_analysis['medic_avg_error']:.2f}")
    print(f"  Heavy average error: {error_analysis['heavy_avg_error']:.2f}")
    print(f"  Engineer average error: {error_analysis['engineer_avg_error']:.2f}")
    
    print(f"\nTolerance Recommendations:")
    tolerance = stats["recommendations"]["tolerance_analysis"]
    print(f"  Current tolerance: {tolerance['current_tolerance']:.1f} units")
    print(f"  Recommended tolerance: {tolerance['recommended_tolerance']:.1f} units")
    print(f"  Rationale: {tolerance['rationale']}")
    
    print(f"\nCoefficient Suggestions:")
    coeffs = stats["recommendations"]["coefficient_suggestions"]
    print(f"  Conservative (5% error): {coeffs['conservative']}")
    print(f"  Moderate (10% error): {coeffs['moderate']} (current)")
    print(f"  Realistic (15% error): {coeffs['realistic']}")

def main():
    """Main analysis function."""
    
    difficulty = "medium"  # Can be changed to normal/hard
    num_seeds = 30
    
    # Run analysis
    stats = analyze_all_seeds(num_seeds, difficulty)
    
    # Print summary
    print_summary(stats)
    
    # Save detailed results
    with open(f'seed_distance_analysis_{difficulty}_{num_seeds}seeds.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nDetailed results saved to: seed_distance_analysis_{difficulty}_{num_seeds}seeds.json")
    
    # Create visualizations
    try:
        create_visualizations(stats)
    except ImportError:
        print("Matplotlib not available, skipping visualizations")
    except Exception as e:
        print(f"Error creating plots: {e}")

if __name__ == "__main__":
    main() 