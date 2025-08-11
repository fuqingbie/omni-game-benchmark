#!/usr/bin/env python3
"""
Analyze Qwen evaluation results and demonstrate media file access.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


def load_evaluation_results(results_path: str) -> Optional[Dict]:
    """Load evaluation results from JSON file."""
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading results: {e}")
        return None


def analyze_media_files(results: Dict) -> None:
    """Analyze and display media file information."""
    print("\n=== MEDIA FILES ANALYSIS ===")
    
    # Check if media files were saved
    if not results.get('media_summary'):
        print("No media files were saved in this evaluation.")
        return
    
    media_summary = results['media_summary']
    output_dir = Path(media_summary['output_directory'])
    
    print(f"Output Directory: {output_dir}")
    print(f"Total Images: {media_summary['total_images']}")
    print(f"Total Audio Files: {media_summary['total_audio_files']}")
    print(f"Total Response Files: {media_summary['total_response_files']}")
    
    # Show how to access files for each step
    print("\n=== STEP-BY-STEP MEDIA FILES ===")
    for step_data in results['steps']:
        step_num = step_data['step']
        media_paths = step_data.get('media_paths', {})
        
        print(f"\nStep {step_num}:")
        print(f"  Command: {step_data['command_desc']}")
        print(f"  Score: {step_data['score_normalized']:.1f}/100")
        
        # Show available media files for this step
        if media_paths.get('image'):
            image_path = output_dir / media_paths['image']
            print(f"  📷 Image: {image_path}")
            print(f"     Exists: {image_path.exists()}")
        
        if media_paths.get('audio'):
            audio_path = output_dir / media_paths['audio']
            print(f"  🔊 Audio: {audio_path}")
            print(f"     Exists: {audio_path.exists()}")
        
        if media_paths.get('response'):
            response_path = output_dir / media_paths['response']
            print(f"  💬 Response: {response_path}")
            print(f"     Exists: {response_path.exists()}")


def show_step_details(results: Dict, step_number: int) -> None:
    """Show detailed information for a specific step."""
    steps = results.get('steps', [])
    
    if step_number < 1 or step_number > len(steps):
        print(f"Invalid step number. Available steps: 1-{len(steps)}")
        return
    
    step_data = steps[step_number - 1]
    media_paths = step_data.get('media_paths', {})
    output_dir = Path(results['media_summary']['output_directory'])
    
    print(f"\n=== STEP {step_number} DETAILS ===")
    print(f"Command: {step_data['command_desc']}")
    print(f"Reward: {step_data['reward']:.2f}")
    print(f"Score: {step_data['score_normalized']:.1f}/100")
    print(f"Objectives Completed: {step_data['objectives_completed']}")
    
    # Show model response if available
    if media_paths.get('response'):
        response_path = output_dir / media_paths['response']
        if response_path.exists():
            print(f"\n--- Model Response ---")
            with open(response_path, 'r', encoding='utf-8') as f:
                response_text = f.read()
                # Show first 200 characters
                if len(response_text) > 200:
                    print(f"{response_text[:200]}...")
                else:
                    print(response_text)
    
    # Show audio data if available
    if media_paths.get('audio'):
        audio_path = output_dir / media_paths['audio']
        if audio_path.exists():
            print(f"\n--- Audio Messages ---")
            with open(audio_path, 'r', encoding='utf-8') as f:
                audio_data = json.load(f)
                for message in audio_data:
                    print(f"  - {message}")


def show_performance_summary(results: Dict) -> None:
    """Show performance summary."""
    print("\n=== PERFORMANCE SUMMARY ===")
    
    config = results['config']
    final_stats = results['final_stats']
    
    print(f"Configuration:")
    print(f"  Difficulty: {config['difficulty']}")
    print(f"  Seed: {config['seed_index']}")
    print(f"  Model: {config['model']}")
    
    print(f"\nResults:")
    print(f"  Final Score: {final_stats['final_score_normalized']:.1f}/100")
    print(f"  Total Steps: {final_stats['total_steps']}")
    print(f"  Objectives Completed: {final_stats['objectives_completed']}/{final_stats['total_objectives']}")
    print(f"  Success Rate: {final_stats['success_rate']:.1%}")
    print(f"  Game Status: {'Completed' if final_stats['terminated'] else 'Truncated'}")


def main():
    """Main analysis function."""
    if len(sys.argv) != 2:
        print("Usage: python analyze_evaluation_results.py <results.json>")
        print("Example: python analyze_evaluation_results.py qwen_eval_medium_seed0_20240101_120000/results.json")
        return
    
    results_path = sys.argv[1]
    results = load_evaluation_results(results_path)
    
    if not results:
        return
    
    # Show performance summary
    show_performance_summary(results)
    
    # Analyze media files
    analyze_media_files(results)
    
    # Interactive step analysis
    while True:
        try:
            print(f"\n=== INTERACTIVE ANALYSIS ===")
            print("Commands:")
            print("  <number>: Show details for step number")
            print("  'q': Quit")
            
            user_input = input("\nEnter command: ").strip()
            
            if user_input.lower() == 'q':
                break
            
            try:
                step_num = int(user_input)
                show_step_details(results, step_num)
            except ValueError:
                print("Invalid input. Please enter a step number or 'q' to quit.")
        
        except KeyboardInterrupt:
            print("\nAnalysis interrupted by user.")
            break
    
    print("\nAnalysis completed!")


if __name__ == "__main__":
    main() 