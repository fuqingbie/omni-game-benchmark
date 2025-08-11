"""
Simplified Cooperative Command Game Environment

A streamlined version with normalized scores built into environment feedback
and simplified logging.
"""

import time
import json
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Fix imports - use absolute imports instead of relative
try:
    from team_member import TeamMember, TeamMemberManager, Position, VoiceAsset
    from game_logic import GameLogic
    from audio_system import AudioSystem
    from visualization import GameVisualization
except ImportError:
    # Create minimal fallback classes if imports fail
    @dataclass
    class Position:
        x: float = 0.0
        y: float = 0.0
    
    @dataclass
    class VoiceAsset:
        text: str
        priority: int = 1
    
    class TeamMember:
        pass
    
    class TeamMemberManager:
        def __init__(self, difficulty="normal", deterministic_commands=True):
            self.members = {}
            self.difficulty = difficulty
            self.deterministic_commands = deterministic_commands
        
        def reset_team(self):
            pass
        
        def update(self, delta_time):
            pass
        
        def get_team_status(self):
            return {"members": {}}
        
        def execute_command(self, member_id, command, position=None):
            return True, "Command executed"
        
        def get_member(self, member_id):
            return None
        
        def get_pending_reports(self):
            return []
    
    class GameLogic:
        def __init__(self, difficulty):
            self.difficulty = difficulty
            self.hidden_objectives = []
        
        def generate_objectives(self):
            return []
        
        def check_objectives(self, mission_objectives, completed_objectives, failed_objectives, team_status):
            return [], [], []
        
        def calculate_main_score(self, completed_objectives, failed_objectives, mission_objectives):
            return 0
        
        def calculate_normalized_score(self, main_score, auxiliary_command_score, completed_objectives, mission_objectives, current_round, optimal_rounds):
            return 0.0
        
        def calculate_final_score(self, completed_objectives, failed_objectives, current_round, max_rounds):
            return 0
        
        def calculate_command_score(self, command, member_id):
            return 0
        
        def check_mission_complete(self, mission_objectives, completed_objectives):
            return False
        
        def check_mission_failed(self, team_status, failed_objectives):
            return False
        
        def get_visible_objectives(self, mission_objectives):
            return mission_objectives
        
        def get_discovery_hints(self, team_status):
            return []
    
    class AudioSystem:
        def __init__(self, assets_dir):
            self.assets_dir = assets_dir
        
        def queue_audio_message(self, voice_asset, member_id, member_role, noise_level=0.0, difficulty="normal"):
            pass
        
        def shutdown(self):
            pass
    
    class GameVisualization:
        def __init__(self):
            pass
        
        def get_render_data(self, team_status, hidden_objectives):
            return {}


class GameDifficulty(Enum):
    """Game difficulty levels"""
    NORMAL = "normal"  # 2 members, simple tasks
    MEDIUM = "medium"  # 4 members, priorities, time limits
    HARD = "hard"      # 6 members, dynamic goals, noisy environment


class GameState(Enum):
    """Current game state"""
    MENU = "menu"
    PLAYING = "playing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GameConfig:
    """Game configuration settings"""
    difficulty: GameDifficulty = GameDifficulty.NORMAL
    max_rounds: Optional[int] = None  # Auto-calculated if None
    enable_audio: bool = True
    enable_tts: bool = True
    noise_level: float = 0.0  # 0.0 to 1.0
    fixed_seed: Optional[int] = None
    seed_index: int = 0  # Index for fixed seed selection (0-9)
    deterministic_commands: bool = True  # If True, all valid commands succeed (default: deterministic)


class CoopCommandEnv:
    """Simplified cooperative command game environment"""
    
    # Fixed seeds for reproducible games
    FIXED_SEEDS = {
        "normal": [12345, 23456, 34567, 45678, 56789, 67890, 78901, 89012, 90123, 1234],
        "medium": [11111, 22222, 33333, 44444, 55555, 66666, 77777, 88888, 99999, 10101],
        "hard": [54321, 65432, 76543, 87654, 98765, 19876, 29876, 39876, 49876, 59876]
    }
    
    def __init__(self, config: Optional[GameConfig] = None, enable_assets: bool = True):
        self.config = config or GameConfig()
        self.state = GameState.MENU
        
        # Round-based tracking
        self.current_round = 0
        self.max_rounds = self.config.max_rounds
        self.optimal_rounds = 0
        self.naive_optimal_rounds = 0
        self.worst_case_optimal_rounds = 0
        
        # Fixed seed handling
        self.current_seed = self._get_game_seed()
        
        # Game components
        self.team_manager = TeamMemberManager(
            difficulty=self.config.difficulty.value,
            deterministic_commands=self.config.deterministic_commands
        )
        self.game_logic = GameLogic(self.config.difficulty)
        self.audio_system = AudioSystem("audio_assets") if self.config.enable_audio else None
        self.visualization = GameVisualization()
        
        # Score tracking
        self.main_score = 0
        self.auxiliary_command_score = 0
        self.mission_objectives = []
        self.completed_objectives = []
        self.failed_objectives = []
        
        # Normalized scoring - calculated once at game start
        self.max_possible_score = 0
        self.score_normalized = 0.0
        
        # Command history
        self.command_history = []
        
        # Audio message queue
        self.pending_audio_messages = []
        
    def _get_game_seed(self) -> int:
        """Get the seed for current game"""
        if self.config.fixed_seed is not None:
            return self.config.fixed_seed
        
        difficulty_key = self.config.difficulty.value
        seed_index = max(0, min(9, self.config.seed_index))
        
        if difficulty_key in self.FIXED_SEEDS:
            return self.FIXED_SEEDS[difficulty_key][seed_index]
        else:
            return self.FIXED_SEEDS["normal"][seed_index]
    
    def _calculate_optimal_rounds(self) -> int:
        """Calculate optimal rounds needed to complete all objectives"""
        if not self.mission_objectives:
            return 10
        
        # Use worst-case optimal rounds as the primary metric
        return self._calculate_worst_case_optimal_rounds()

    def _calculate_naive_optimal_rounds(self) -> int:
        """Calculate optimal rounds assuming perfect knowledge (no discovery needed)"""
        if not self.mission_objectives:
            return 10
        
        # Naive assumption: 1 round per objective with perfect knowledge
        return len(self.mission_objectives)

    def _calculate_worst_case_optimal_rounds(self) -> int:
        """Calculate optimal rounds considering hidden objective discovery complexity"""
        if not self.mission_objectives:
            return 10
        
        visible_objectives = self.game_logic.get_visible_objectives(self.mission_objectives)
        hidden_objectives = self.game_logic.hidden_objectives
        
        # Rounds for visible objectives (straightforward completion)
        visible_rounds = len(visible_objectives)
        
        # Rounds for hidden objective discovery and completion
        hidden_discovery_rounds = 0
        if hidden_objectives:
            # Discovery complexity calculation
            for hidden_obj in hidden_objectives:
                discovery_rounds = self._estimate_discovery_rounds(hidden_obj)
                completion_rounds = 1  # 1 round to complete after discovery
                hidden_discovery_rounds += discovery_rounds + completion_rounds
        
        # Add exploration overhead for systematic search
        map_size = 100  # 100x100 map
        scout_count = self._count_scouts()
        exploration_overhead = self._calculate_exploration_overhead(len(hidden_objectives), scout_count, map_size)
        
        total_rounds = visible_rounds + hidden_discovery_rounds + exploration_overhead
        
        # Cap at reasonable maximum to prevent excessive rounds
        max_reasonable = len(self.mission_objectives) * 3
        return min(total_rounds, max_reasonable)

    def _estimate_discovery_rounds(self, hidden_obj: Dict) -> int:
        """Estimate rounds needed to discover a single hidden objective"""
        # Discovery probability: Scout 80%, Others 40%
        scout_discovery_prob = 0.8
        other_discovery_prob = 0.4
        
        scout_count = self._count_scouts()
        other_count = max(1, len(self.team_manager.members) - scout_count)
        
        # Expected rounds for discovery assuming optimal positioning
        # Using geometric distribution: E[X] = 1/p for probability p
        if scout_count > 0:
            # Use scout probability (better discovery rate)
            expected_rounds = 1 / scout_discovery_prob  # ~1.25 rounds
        else:
            # Use other member probability
            expected_rounds = 1 / other_discovery_prob  # ~2.5 rounds
        
        # Reduce positioning rounds for budget constraints
        positioning_rounds = 1  # Reduced from 2 (assumes better movement planning)
        
        return max(1, int(expected_rounds + positioning_rounds))

    def _calculate_exploration_overhead(self, hidden_count: int, scout_count: int, map_size: int) -> int:
        """Calculate additional rounds needed for systematic exploration"""
        if hidden_count == 0:
            return 0
        
        # Grid search strategy: divide map into search zones
        discovery_radius = 15.0  # From game_logic.py
        zone_coverage = discovery_radius * 2  # Effective coverage per position
        zones_per_axis = int(map_size / zone_coverage)
        total_zones = zones_per_axis * zones_per_axis
        
        # Exploration efficiency based on team composition (more optimistic for budget)
        if scout_count >= 2:
            exploration_efficiency = 0.2  # Good scouting capability
        elif scout_count == 1:
            exploration_efficiency = 0.3  # Moderate scouting (reduced from 0.5)
        else:
            exploration_efficiency = 0.5  # Poor scouting (reduced from 0.8)
        
        # Estimate zones to search before finding all hidden objectives
        zones_to_search = int(total_zones * exploration_efficiency)
        
        # Rounds per zone (movement + search)
        rounds_per_zone = 1
        
        # Parallel exploration with multiple scouts
        effective_scout_count = max(1, scout_count)
        exploration_rounds = zones_to_search // effective_scout_count
        
        return max(1, exploration_rounds)

    def _count_scouts(self) -> int:
        """Count number of scout team members"""
        scout_count = 0
        team_status = self.team_manager.get_team_status()
        for member_info in team_status.get("members", {}).values():
            if member_info.get("role") == "scout":
                scout_count += 1
        return scout_count
    
    def _calculate_max_possible_score(self) -> int:
        """Calculate theoretical maximum score for normalization"""
        if not self.mission_objectives or not self.optimal_rounds or not self.max_rounds:
            # Default estimates by difficulty
            base_scores = {'normal': 300, 'medium': 600, 'hard': 900}
            return base_scores.get(self.config.difficulty.value, 600)
        
        # Base score from all objectives
        base_score = sum(obj.get("score_value", 0) for obj in self.mission_objectives)
        
        # Maximum efficiency bonuses (theoretical perfect performance)
        if self.max_rounds > 0:
            rounds_bonus = int(base_score * (1.0 - self.optimal_rounds / self.max_rounds) * 0.5)
        else:
            rounds_bonus = 0
        
        efficiency_bonus = int(base_score * 1.0 * 0.3)  # 100% command success
        
        # Dynamic task bonus for hard mode
        dynamic_bonus = 200 if self.config.difficulty.value == "hard" else 0
        
        max_score = base_score + rounds_bonus + efficiency_bonus + dynamic_bonus
        return max(max_score, 1)
    
    def _update_normalized_score(self):
        """Update normalized score (0-100 range)"""
        if self.max_possible_score > 0:
            self.score_normalized = min(100.0, max(0.0, 
                (self.main_score / self.max_possible_score) * 100))
        else:
            self.score_normalized = 0.0
    
    def start_game(self) -> Dict[str, Any]:
        """Start a new game session"""
        self.state = GameState.PLAYING
        self.current_round = 0
        self.main_score = 0
        self.auxiliary_command_score = 0
        self.score_normalized = 0.0
        
        # Set random seed for reproducible games
        random.seed(self.current_seed)
        
        # Initialize mission objectives
        self.mission_objectives = self.game_logic.generate_objectives()
        self.completed_objectives = []
        self.failed_objectives = []
        
        # Reset team state
        self.team_manager.reset_team()
        self.command_history = []
        
                # Calculate game parameters and max possible score
        self.optimal_rounds = self._calculate_optimal_rounds()
        self.naive_optimal_rounds = self._calculate_naive_optimal_rounds()
        self.worst_case_optimal_rounds = self._calculate_worst_case_optimal_rounds()
        
        if self.config.max_rounds is None:
            # Cap at 30 for budget constraints while keeping reasonable ratios
            self.max_rounds = min(3 * self.optimal_rounds, 30)
        else:
            self.max_rounds = self.config.max_rounds

        self.max_possible_score = self._calculate_max_possible_score()
        
        return self.get_game_state()
    
    def pause_game(self) -> Dict[str, Any]:
        """Pause/unpause the game"""
        if self.state == GameState.PLAYING:
            self.state = GameState.PAUSED
        elif self.state == GameState.PAUSED:
            self.state = GameState.PLAYING
        return self.get_game_state()
    
    def end_game(self) -> Dict[str, Any]:
        """End the current game session"""
        self.state = GameState.COMPLETED
        
        # Calculate final score with bonuses/penalties
        final_score = self.game_logic.calculate_final_score(
            self.completed_objectives, 
            self.failed_objectives,
            self.current_round,
            self.max_rounds
        )
        
        # Update main score if final calculation is higher
        if final_score > self.main_score:
            self.main_score = final_score
        
        # Update normalized score
        self._update_normalized_score()
        
        return self.get_game_state()
    
    def execute_command(self, command: str, target_member: Optional[str] = None, 
                       target_position: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """Execute a command for a team member"""
        if self.state != GameState.PLAYING:
            return {"success": False, "response": "Game not in progress", "member_id": None}
        
        # Find target member or use first available
        if target_member:
            member_id = target_member
        else:
            team_status = self.team_manager.get_team_status()
            member_ids = list(team_status["members"].keys())
            member_id = member_ids[0] if member_ids else None
        
        if not member_id:
            return {"success": False, "response": "No team members available", "member_id": None}
        
        # Convert tuple to Position object if needed for team manager
        position_for_manager = None
        if target_position is not None:
            if isinstance(target_position, tuple) and len(target_position) == 2:
                from team_member import Position
                position_for_manager = Position(float(target_position[0]), float(target_position[1]))
            else:
                position_for_manager = target_position
        
        # Execute command through team manager
        success, response = self.team_manager.execute_command(member_id, command, position_for_manager)
        
        # Record command
        self.command_history.append({
            "round": self.current_round,
            "command": command,
            "target_member": target_member,
            "target_position": target_position,
            "success": success,
            "response": response
        })
        
        # Update auxiliary score for successful commands
        if success:
            self.auxiliary_command_score += self.game_logic.calculate_command_score(command, member_id)
        
        # Queue audio response and add to pending messages for observation
        if self.audio_system and response:
            member = self.team_manager.get_member(member_id)
            if member:
                voice_asset = VoiceAsset(text=response, priority=3)
                self.audio_system.queue_audio_message(
                    voice_asset, member_id, member.role.value,
                    noise_level=self.config.noise_level,
                    difficulty=self.config.difficulty.value
                )
                
                # Add to pending messages for gym wrapper observation
                self.pending_audio_messages.append({
                    "member_id": member_id,
                    "message": response,
                    "timestamp": self.current_round,
                    "priority": 3,
                    "type": "command_response"
                })
        
        return {
            "success": success,
            "response": response,
            "member_id": member_id,
            "score_change": self.game_logic.calculate_command_score(command, member_id) if success else 0
        }
    
    def step(self) -> Dict[str, Any]:
        """Execute one round of the game"""
        if self.state != GameState.PLAYING:
            return self.get_game_state()
        
        self.current_round += 1
        
        # Check if max rounds reached
        if self.current_round >= self.max_rounds:
            return self.end_game()
        
        # Update team members
        self.team_manager.update(1.0)
        
        # Process team member reports
        reports = self.team_manager.get_pending_reports()
        for report in reports:
            if self.audio_system:
                # Properly unpack the tuple: (member_id, voice_asset)
                member_id, voice_asset = report
                
                # Get member role from team manager
                member_role = "unknown"
                if member_id in self.team_manager.members:
                    member_role = self.team_manager.members[member_id].role.value
                
                voice_asset_copy = VoiceAsset(text=voice_asset.text, priority=2)
                self.audio_system.queue_audio_message(
                    voice_asset_copy, member_id, member_role,
                    noise_level=self.config.noise_level,
                    difficulty=self.config.difficulty.value
                )
                
                # Add to pending messages for gym wrapper observation
                self.pending_audio_messages.append({
                    "member_id": member_id,
                    "message": voice_asset.text,
                    "timestamp": self.current_round,
                    "priority": 2,
                    "type": "member_report"
                })
        
        # Check objective completion and hidden objective discovery
        team_status = self.team_manager.get_team_status()
        new_completed, new_failed, newly_discovered = self.game_logic.check_objectives(
            self.mission_objectives, 
            self.completed_objectives, 
            self.failed_objectives,
            team_status
        )
        
        # Process newly discovered objectives
        for objective in newly_discovered:
            if self.audio_system:
                discoverer_id = objective.get("discovered_by", "Unknown")
                voice_asset = VoiceAsset(
                    text=f"New objective discovered: {objective['description']}", 
                    priority=4
                )
                self.audio_system.queue_audio_message(
                    voice_asset, discoverer_id, "discovery",
                    noise_level=self.config.noise_level,
                    difficulty=self.config.difficulty.value
                )
                
                # Add to pending messages for gym wrapper observation
                self.pending_audio_messages.append({
                    "member_id": discoverer_id,
                    "message": f"New objective discovered: {objective['description']}",
                    "timestamp": self.current_round,
                    "priority": 4,
                    "type": "discovery"
                })
        
        # Update objective lists
        self.completed_objectives.extend(new_completed)
        self.failed_objectives.extend(new_failed)
        # Note: newly_discovered objectives are already in mission_objectives
        # They were just status-updated from "hidden" to "pending", so no need to add them again
        
        # Update score
        self.main_score = self.game_logic.calculate_main_score(
            self.completed_objectives, self.failed_objectives, self.mission_objectives
        )
        
        # Update normalized score based on current progress
        self.score_normalized = self.game_logic.calculate_normalized_score(
            self.main_score, self.auxiliary_command_score,
            self.completed_objectives, self.mission_objectives,
            self.current_round, self.optimal_rounds
        )
        
        # Check win/lose conditions
        if self.game_logic.check_mission_complete(self.mission_objectives, self.completed_objectives):
            self.state = GameState.COMPLETED
        elif self.game_logic.check_mission_failed(self.team_manager.get_team_status(), self.failed_objectives):
            self.state = GameState.FAILED
        
        return self.get_game_state()
    
    def get_game_state(self) -> Dict[str, Any]:
        """Get current game state with normalized scores in feedback"""
        team_status = self.team_manager.get_team_status()
        
        # Clear old audio messages (keep only last 5)
        if len(self.pending_audio_messages) > 5:
            self.pending_audio_messages = self.pending_audio_messages[-5:]
        
        # Get only visible objectives (not hidden)
        visible_objectives = self.game_logic.get_visible_objectives(self.mission_objectives)
        
        # Get discovery hints for nearby hidden objectives
        discovery_hints = self.game_logic.get_discovery_hints(team_status)
        
        return {
            "state": self.state.value,
            "main_score": self.main_score,
            "auxiliary_command_score": self.auxiliary_command_score,
            # Normalized score included in main feedback
            "score_normalized": round(self.score_normalized, 1),
            "max_possible_score": self.max_possible_score,
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "optimal_rounds": self.optimal_rounds,
            "naive_optimal_rounds": self.naive_optimal_rounds,
            "worst_case_optimal_rounds": self.worst_case_optimal_rounds,
            "rounds_remaining": max(0, self.max_rounds - self.current_round),
            "game_seed": self.current_seed,
            "team_status": team_status,
            "mission_objectives": visible_objectives,  # Only show visible objectives
            "total_objectives_count": len(self.mission_objectives),  # Total count for progress tracking
            "hidden_objectives_count": len(self.game_logic.hidden_objectives),  # Number still hidden
            "completed_objectives": self.completed_objectives,
            "failed_objectives": self.failed_objectives,
            "discovery_hints": discovery_hints,  # Hints about nearby hidden objectives
            "recent_commands": self.command_history[-3:],  # Last 3 commands
            "audio_messages": self.pending_audio_messages,
            "visualization_data": self.visualization.get_render_data(team_status, self.game_logic.hidden_objectives)
        }
    
    def get_audio_messages(self) -> List[Dict]:
        """Get pending audio messages"""
        messages = self.pending_audio_messages.copy()
        self.pending_audio_messages.clear()
        return messages
    
    def shutdown(self):
        """Clean shutdown"""
        if self.audio_system:
            self.audio_system.shutdown()
        self.state = GameState.MENU
    
    def get_evaluation_metrics(self) -> Dict[str, Any]:
        """Get simplified evaluation metrics"""
        return {
            "main_score": self.main_score,
            "score_normalized": self.score_normalized,
            "max_possible_score": self.max_possible_score,
            "rounds_used": self.current_round,
            "max_rounds": self.max_rounds,
            "optimal_rounds": self.optimal_rounds,
            "naive_optimal_rounds": self.naive_optimal_rounds,
            "worst_case_optimal_rounds": self.worst_case_optimal_rounds,
            "objectives_completed": len(self.completed_objectives),
            "total_objectives": len(self.mission_objectives),
            "success_rate": len(self.completed_objectives) / len(self.mission_objectives) if self.mission_objectives else 0.0,
            "efficiency": min(100, (self.optimal_rounds / max(1, self.current_round)) * 100) if self.current_round > 0 else 0,
            "game_completed": self.state in [GameState.COMPLETED, GameState.FAILED],
            "difficulty": self.config.difficulty.value,
            "seed_index": self.config.seed_index
        }
    
    @classmethod
    def create_fixed_seed_game(cls, difficulty: GameDifficulty, seed_index: int, 
                              max_rounds: Optional[int] = None) -> 'CoopCommandEnv':
        """Create a game with fixed seed for reproducible testing"""
        config = GameConfig(
            difficulty=difficulty,
            max_rounds=max_rounds,
            seed_index=seed_index,
            enable_audio=True
        )
        return cls(config=config, enable_assets=True)