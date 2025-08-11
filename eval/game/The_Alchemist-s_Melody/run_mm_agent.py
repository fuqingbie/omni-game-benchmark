#!/usr/bin/env python3
"""
Sound Alchemist Multi-Modal Agent Test Script
"""
import time
import numpy as np
import sys
import os
from typing import Dict, Any

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# Import configuration file
from config import Config, initialize_config

# Fix import statements - use absolute imports
from game.sound_alchemist.sound_alchemist_env import SoundAlchemistEnv, COLOR_ID_MAP
from game.sound_alchemist.multimodal_agent import MultimodalAgent

def print_observation_info(obs: Dict[str, Any]):
    """Print a summary of the observation information"""
    image = obs["image"]
    audio = obs["audio"]
    state = obs["state"]
    
    # Image statistics
    unique_colors = len(np.unique(image.reshape(-1, image.shape[-1]), axis=0))
    brightness = np.mean(image)
    
    print(f"  Image: {image.shape}, brightness={brightness:.1f}, unique_colors={unique_colors}")
    print(f"  State: score={state[0]:.0f}, lives={state[1]:.0f}, solved={state[2]:.0f}, tick={state[3]:.0f}")

def main():
    # Initialize the configuration system
    if not initialize_config():
        print("Configuration initialization failed, exiting program")
        return
    
    print("=== Sound Alchemist Multi-Modal Agent Test ===")
    
    # Create the environment using the configuration
    env = SoundAlchemistEnv(
        difficulty=Config.DEFAULT_DIFFICULTY,
        save_data=Config.Environment.SAVE_DATA,
        save_sequence=Config.Environment.SAVE_SEQUENCE,
        save_dir=Config.Environment.SAVE_DIR
    )
    
    # Create the agent using the configuration
    agent = MultimodalAgent(
        verbose=Config.Agent.VERBOSE, 
        use_local_fallback=Config.Agent.USE_LOCAL_FALLBACK,
        conversation_strategy=Config.Agent.CONVERSATION_STRATEGY,
        max_retries=Config.Agent.MAX_RETRIES
    )
    
    # Connect the agent to the environment so it can get game state information
    agent.set_game_environment(env)
    
    try:
        # Run multiple episodes
        num_episodes = Config.Experiment.NUM_EPISODES
        
        for episode in range(num_episodes):
            print(f"\n{'='*50}")
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"{'='*50}")
            
            # Reset the agent's state - use the existing method or reset manually
            if hasattr(agent, 'reset_for_new_episode'):
                agent.reset_for_new_episode(episode + 1)
            else:
                # Manually reset the agent's episode-related state
                agent.current_episode = episode + 1
                agent.current_step = 0
                agent.current_round_start_time = time.time()
                agent.last_game_completed = False
                # Reset the memory manager
                if hasattr(agent.memory_manager, 'reset_for_new_episode'):
                    agent.memory_manager.reset_for_new_episode(episode + 1)
                print(f"Agent manually reset for episode {episode + 1}")
            
            obs, info = env.reset()
            episode_reward = 0
            step_count = 0
            max_steps = Config.Environment.MAX_STEPS_PER_EPISODE
            game_completed = False  
            
            print("Initial observation:")
            print_observation_info(obs)
            
            # Display the available color blocks in the current game
            if hasattr(env, 'get_current_available_colors'):
                available_colors = env.get_current_available_colors()
                print(f"Available color blocks this round: {available_colors}")
            
            while step_count < max_steps and not game_completed:
                print(f"\n--- Step {step_count + 1} ---")
                
                # Agent chooses an action
                action, end_game = agent.act(obs)
                if end_game:
                    game_completed = True
                    print("Game ended.")
                    break

                color_name = list(COLOR_ID_MAP.keys())[action]
                print(f"Agent chose action {action} ({color_name})")
                
                # Get the agent's text output
                current_text = agent.get_current_step_text()
                if current_text:
                    # If the environment supports setting step text, set it
                    if hasattr(env, 'set_step_text'):
                        env.set_step_text(current_text)
                    print(f"Agent reasoning: {current_text[:100]}..." if len(current_text) > 100 else f"Agent reasoning: {current_text}")
                
                # Display the most recent model output and thought process
                if agent.model_output_history:
                    latest_output = agent.model_output_history[-1]
                    if latest_output.get("success"):
                        response_time = latest_output.get("response_time", 0)
                        response_text = latest_output.get("response_text", "")
                        print(f"Model response time: {response_time:.2f}s, length: {len(response_text)} chars")
                        
                        # Display key parts of the thought process
                        if response_text and len(response_text) > 50:
                            # Try to extract the decision part
                            decision_start = response_text.upper().find("DECISION:")
                            if decision_start >= 0:
                                decision_text = response_text[decision_start:decision_start+200]
                                print(f"Model reasoning: {decision_text}...")
                            else:
                                # If the DECISION tag is not found, display the last 100 characters
                                print(f"Model thinking: ...{response_text[-100:]}")

                # Display the current memory state
                memory_stats = agent.memory_manager.get_conversation_context_for_api("hybrid")
                if step_count > 0:
                    print(f"Memory context length: {len(memory_stats)} chars")

                try:
                    # Execute the action
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    step_count += 1
                    
                    print(f"Reward: {reward}, Total: {episode_reward}")
                    
                    # Check the game completion status
                    current_solved = float(obs["state"][2])
                    current_score = float(obs["state"][0])
                    
                    # Multiple ways to detect game completion
                    if current_solved > 0:
                        game_completed = True
                        print("🎉 GAME COMPLETED! Sequence successfully matched!")
                    elif reward >= Config.Scoring.COMPLETION_MULTIPLIER:
                        game_completed = True
                        print("🎉 GAME COMPLETED! High reward achieved!")
                    elif "game_completed" in info and info["game_completed"]:
                        game_completed = True
                        print("🎉 GAME COMPLETED! Environment confirmed completion!")
                    
                    # If the game is completed, settle the score
                    if game_completed:
                        print(f"\n{'='*40}")
                        print(f"🏆 ROUND COMPLETION SUMMARY 🏆")
                        print(f"{'='*40}")
                        print(f"Final Score: {current_score}")
                        print(f"Total Steps: {step_count}")
                        print(f"Episode Reward: {episode_reward}")
                        print(f"Efficiency: {current_score/step_count:.2f} points/step")
                        
                        # Get the agent's completion statistics - in a safe way
                        if hasattr(agent, 'get_completion_stats'):
                            completion_stats = agent.get_completion_stats()
                            print(f"Agent Completions: {completion_stats['total_completions']}")
                            if completion_stats['total_completions'] > 0:
                                print(f"Average Completion Score: {completion_stats['average_score']:.1f}")
                                print(f"Best Score: {completion_stats['best_score']}")
                                print(f"Completion Rate: {completion_stats['completion_rate']:.1f}%")
                        
                        # Get the learning progress summary - in a safe way
                        if hasattr(agent, 'get_learning_progress_summary'):
                            learning_summary = agent.get_learning_progress_summary()
                            print(f"\n{learning_summary}")
                        
                        # Provide positive feedback to the agent
                        agent.update_color_feedback(color_name, True)
                        
                        # Display the sequence completion information
                        if hasattr(env, '_build_detailed_game_state'):
                            try:
                                detailed_state = env._build_detailed_game_state(action=None, is_reset=False)
                                completed_sequence = detailed_state.get("current_correct_sequence", [])
                                if completed_sequence:
                                    print(f"Completed Sequence: {' → '.join(completed_sequence)}")
                            except Exception as e:
                                print(f"Could not get sequence details: {e}")
                        
                        # Manually record completion information (if the agent does not handle it automatically)
                        if not hasattr(agent, 'game_completion_history'):
                            agent.game_completion_history = []
                        
                        completion_record = {
                            "timestamp": time.time(),
                            "episode": episode + 1,
                            "final_score": current_score,
                            "steps_taken": step_count,
                            "episode_reward": episode_reward,
                            "efficiency": current_score / step_count if step_count > 0 else 0
                        }
                        agent.game_completion_history.append(completion_record)
                        
                        break
                    
                    # Display important information from the game state
                    if "needs_restart_from_beginning" in info:
                        print(f"Needs restart: {info['needs_restart_from_beginning']}")
                    if "sequence_reset" in info:
                        print(f"Sequence reset: {info['sequence_reset']}")
                    
                    if reward > 0 and not game_completed:
                        print("✓ Positive reward achieved!")
                        # Provide color feedback to the agent
                        agent.update_color_feedback(color_name, True)
                    elif reward <= 0:
                        agent.update_color_feedback(color_name, False)
                    
                    print_observation_info(obs)
                    
                    # Display memory statistics every 5 steps
                    if step_count % 5 == 0:
                        rounds_count = len(agent.memory_manager.conversation_rounds) if hasattr(agent, 'memory_manager') else 0
                        summaries_count = len(agent.memory_manager.compressed_summaries) if hasattr(agent, 'memory_manager') and hasattr(agent.memory_manager, 'compressed_summaries') else 0
                        
                        # Safely get model statistics
                        if hasattr(agent, 'get_model_output_stats'):
                            model_stats = agent.get_model_output_stats()
                            print(f"📊 Memory: {rounds_count} active rounds, {summaries_count} compressed summaries")
                            print(f"📈 Model: {model_stats['successful_calls']}/{model_stats['total_calls']} calls, "
                                  f"avg response time: {model_stats.get('avg_response_time', 0):.2f}s")
                        else:
                            print(f"📊 Memory: {rounds_count} active rounds, {summaries_count} compressed summaries")
                            print(f"📈 Model: Statistics not available")
                    
                    # Check if the environment ended naturally
                    if terminated or truncated:
                        print(f"Episode ended: terminated={terminated}, truncated={truncated}")
                        
                        # If the episode ended but the game was not completed, check if it was a failure
                        if not game_completed:
                            lives_remaining = float(obs["state"][1])
                            if lives_remaining <= 0:
                                print("💀 Game Over - No lives remaining")
                            else:
                                print("⏰ Episode ended - Time/step limit reached")
                        
                        break
                        
                except Exception as e:
                    print(f"Error during step {step_count}: {e}")
                    import traceback
                    traceback.print_exc()
                    break
        
            # Display the learned mappings
            if agent.learned_color_note_mapping:
                print(f"  Learned mappings: {agent.learned_color_note_mapping}")
            
            time.sleep(1)  # Short pause
        
        # Final summary after all episodes are finished
        print(f"\n{'='*60}")
        print("🏁 FINAL GAME COMPLETION SUMMARY")
        print(f"{'='*60}")
        
        # Display the game scoring mechanism
        if hasattr(env, 'get_current_game_scoring_mechanism'):
            scoring_mechanism = env.get_current_game_scoring_mechanism()
            print(f"\n📋 Current Game Scoring Mechanism:")
            print(f"{'='*50}")
            
            # Original game scoring
            game_scoring = scoring_mechanism["game_scoring"]
            print(f"🎮 Original Game Scoring:")
            print(f"  Base Score: {game_scoring['base_score']}")
            print(f"  Difficulty Multipliers: {game_scoring['difficulty_multipliers']}")
            print(f"  Perfect Play Bonus: {game_scoring['bonuses']['perfect_play']}")
            print(f"  Sequence Bonus: {game_scoring['bonuses']['sequence_bonus']}")
            print(f"  Mistake Penalty: {game_scoring['penalties']['mistake_penalty']}")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure data is saved
        try:
            env.close()
        except:
            pass

if __name__ == "__main__":
    main()