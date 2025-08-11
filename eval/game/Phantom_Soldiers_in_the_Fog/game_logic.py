"""
Game Logic System

Handles command parsing, objective generation, mission scoring, and game state evaluation.
"""

import random
import re
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

try:
    from .team_member import Position, TeamMemberRole
except ImportError:
    from team_member import Position, TeamMemberRole


class ObjectiveType(Enum):
    """Types of mission objectives"""
    MOVE_TO_POSITION = "move_to_position"
    SECURE_AREA = "secure_area"
    EXTRACT_TARGET = "extract_target"
    DEFEND_POSITION = "defend_position"
    RECONNAISSANCE = "reconnaissance"
    RESUPPLY = "resupply"


class ObjectiveStatus(Enum):
    """Status of mission objectives"""
    HIDDEN = "hidden"        # Hidden objective - not visible to player
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class MissionObjective:
    """Individual mission objective"""
    obj_id: str
    obj_type: ObjectiveType
    description: str
    target_position: Optional[Position] = None
    required_members: List[str] = None
    time_limit: Optional[int] = None  # seconds
    score_value: int = 100
    status: ObjectiveStatus = ObjectiveStatus.PENDING
    assigned_members: List[str] = None
    start_time: Optional[float] = None
    # Hidden objective properties
    discovery_position: Optional[Position] = None  # Position where objective can be discovered
    discovery_radius: float = 15.0  # Radius for discovery
    discovered_by: Optional[str] = None  # Which team member discovered it


class GameLogic:
    """Core game logic and rules engine"""
    
    def __init__(self, difficulty):
        self.difficulty = difficulty.value if hasattr(difficulty, 'value') else difficulty
        self.current_objectives = []
        self.hidden_objectives = []  # List of hidden objectives
        self.command_patterns = self._build_command_patterns()
        
        # Difficulty settings
        self.difficulty_config = self._get_difficulty_config(self.difficulty)
        
        # Game state tracking
        self.total_commands_issued = 0
        self.successful_commands = 0
        self.mission_start_time = None
        
        # Hidden objectives discovery system
        self.exploration_zones_checked = set()  # Track explored areas
    
    def _get_difficulty_config(self, difficulty: str) -> Dict:
        """Get configuration settings based on difficulty"""
        configs = {
            "normal": {
                "team_size": 2,
                "objective_count": 3,
                "hidden_objective_ratio": 0.3,  # 30% of objectives are hidden
                "time_pressure": False,
                "conflicting_info": False,
                "dynamic_objectives": False,
                "noise_level": 0.0,
                "movement_error": 1.0  # Low movement error
            },
            "medium": {
                "team_size": 4,
                "objective_count": 5,
                "hidden_objective_ratio": 0.4,  # 40% of objectives are hidden
                "time_pressure": True,
                "conflicting_info": True,
                "dynamic_objectives": False,
                "noise_level": 0.2,
                "movement_error": 2.0  # Medium movement error
            },
            "hard": {
                "team_size": 6,
                "objective_count": 7,
                "hidden_objective_ratio": 0.5,  # 50% of objectives are hidden
                "time_pressure": True,
                "conflicting_info": True,
                "dynamic_objectives": True,
                "noise_level": 0.4,
                "movement_error": 3.0  # High movement error
            }
        }
        return configs.get(difficulty, configs["normal"])
    
    def _build_command_patterns(self) -> Dict[str, Dict]:
        """Build regex patterns for command recognition"""
        return {
            "move": {
                "patterns": [
                    r"move (?:to )?(?:position )?(?P<x>-?\d+)[,\s]+(?P<y>-?\d+)",
                    r"go (?:to )?(?P<x>-?\d+)[,\s]+(?P<y>-?\d+)",
                    r"advance (?:to )?(?:position )?(?P<x>-?\d+)[,\s]+(?P<y>-?\d+)",
                    r"relocate (?:to )?(?P<x>-?\d+)[,\s]+(?P<y>-?\d+)"
                ],
                "priority": 1
            },
            "attack": {
                "patterns": [
                    r"attack (?:position )?(?P<x>-?\d+)[,\s]+(?P<y>-?\d+)",
                    r"engage (?:target )?(?:at )?(?P<x>-?\d+)[,\s]+(?P<y>-?\d+)",
                    r"fire (?:at )?(?P<x>-?\d+)[,\s]+(?P<y>-?\d+)",
                    r"take out (?:target )?(?:at )?(?P<x>-?\d+)[,\s]+(?P<y>-?\d+)"
                ],
                "priority": 3
            },
            "defend": {
                "patterns": [
                    r"defend (?:position )?(?P<x>-?\d+)[,\s]+(?P<y>-?\d+)",
                    r"hold (?:position )?(?P<x>-?\d+)[,\s]+(?P<y>-?\d+)",
                    r"secure (?:area )?(?P<x>-?\d+)[,\s]+(?P<y>-?\d+)",
                    r"maintain (?:position )?(?P<x>-?\d+)[,\s]+(?P<y>-?\d+)"
                ],
                "priority": 2
            },
            "recon": {
                "patterns": [
                    r"scout (?:area )?(?P<x>-?\d+)[,\s]+(?P<y>-?\d+)",
                    r"reconnaissance (?:at )?(?P<x>-?\d+)[,\s]+(?P<y>-?\d+)",
                    r"survey (?:area )?(?P<x>-?\d+)[,\s]+(?P<y>-?\d+)",
                    r"check (?:position )?(?P<x>-?\d+)[,\s]+(?P<y>-?\d+)"
                ],
                "priority": 1
            },
            "support": {
                "patterns": [
                    r"support (?P<target>\w+)",
                    r"assist (?P<target>\w+)",
                    r"help (?P<target>\w+)",
                    r"cover (?P<target>\w+)"
                ],
                "priority": 2
            },
            "status": {
                "patterns": [
                    r"(?:report )?status",
                    r"sitrep",
                    r"what'?s your status",
                    r"check in"
                ],
                "priority": 1
            }
        }
    
    def parse_command(self, command: str, target_member: Optional[str] = None, 
                     target_position: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """Parse natural language command into structured format"""
        command = command.lower().strip()
        
        # Find matching command pattern
        parsed_command = None
        extracted_position = None
        command_type = None
        
        for cmd_type, cmd_info in self.command_patterns.items():
            for pattern in cmd_info["patterns"]:
                match = re.search(pattern, command)
                if match:
                    parsed_command = cmd_type
                    command_type = cmd_type
                    
                    # Extract position if present in command
                    groups = match.groupdict()
                    if "x" in groups and "y" in groups:
                        try:
                            extracted_position = (float(groups["x"]), float(groups["y"]))
                        except ValueError:
                            pass
                    
                    break
            
            if parsed_command:
                break
        
        if not parsed_command:
            return {
                "success": False,
                "message": "Command not recognized. Please use clear military commands.",
                "suggestions": [
                    "move to 10, 5",
                    "attack position 15, 20", 
                    "defend area 8, 12",
                    "report status"
                ]
            }
        
        # Use extracted position or provided target position
        final_position = extracted_position or target_position
        
        # Validate position bounds if provided
        if final_position:
            x, y = final_position
            if x < 0 or x > 100 or y < 0 or y > 100:
                return {
                    "success": False,
                    "message": f"Position ({x}, {y}) is outside map bounds (0-100). Please choose coordinates within the map.",
                    "suggestions": [
                        f"{command_type} to {max(0, min(100, x))}, {max(0, min(100, y))}",
                        "Try coordinates between 0 and 100"
                    ]
                }
        
        if not final_position and command_type in ["move", "attack", "defend", "recon"]:
            return {
                "success": False,
                "message": "Command requires a target position. Example: 'move to 10, 15'",
                "suggestions": [
                    f"{command_type} to 10, 15",
                    f"{command_type} position 20, 25"
                ]
            }
        
        # If no target member specified, try to find best match
        if not target_member:
            target_member = self._suggest_team_member(command_type, final_position)
        
        return {
            "success": True,
            "command": parsed_command,
            "command_type": command_type,
            "member_id": target_member,
            "target_position": final_position,
            "priority": self.command_patterns[command_type]["priority"]
        }
    
    def _suggest_team_member(self, command_type: str, position: Optional[Tuple[float, float]]) -> str:
        """Suggest a team member for a command if not specified"""
        # Default team member names based on difficulty
        difficulty_teams = {
            "normal": ["scout_1", "heavy_1"],
            "medium": ["scout_1", "heavy_1", "medic_1", "engineer_1"],
            "hard": ["scout_1", "heavy_1", "medic_1", "engineer_1", "sniper_1", "support_1"]
        }
        
        available_members = difficulty_teams.get(self.difficulty, ["scout_1", "heavy_1"])
        
        # Simple role-based assignment
        if command_type in ["reconnaissance", "recon"]:
            return "scout_1" if "scout_1" in available_members else available_members[0]
        elif command_type in ["attack"]:
            return "heavy_1" if "heavy_1" in available_members else available_members[0]
        else:
            # Default to first available member
            return available_members[0]
    
    def calculate_command_score(self, command: str, member_id: str) -> int:
        """Calculate score for successful command execution"""
        base_score = 10
        
        # Parse command to get type
        parsed = self.parse_command(command)
        if parsed["success"]:
            command_type = parsed["command_type"]
            priority = parsed["priority"]
            
            # Higher priority commands give more points
            score = base_score * priority
            
            # Bonus for well-formed commands
            if parsed.get("target_position"):
                score += 5
            
            return score
        
        return base_score
    
    def generate_objectives(self) -> List[Dict]:
        """Generate mission objectives based on difficulty"""
        # Clear any existing hidden objectives to prevent accumulation
        self.hidden_objectives.clear()
        
        objectives = []
        config = self.difficulty_config
        
        # Generate primary objectives
        for i in range(config["objective_count"]):
            obj_type = self._select_objective_type()
            objective = self._create_objective(f"obj_{i+1}", obj_type)
            objectives.append(objective)
        
        # Mark some objectives as hidden based on difficulty
        hidden_count = int(len(objectives) * config.get("hidden_objective_ratio", 0))
        hidden_indices = random.sample(range(len(objectives)), min(hidden_count, len(objectives)))
        
        for idx in hidden_indices:
            objectives[idx]["status"] = "hidden"
            # Create discovery zone near objective
            target_pos = objectives[idx]["target_position"]
            discovery_offset_x = random.uniform(-20, 20)
            discovery_offset_y = random.uniform(-20, 20)
            objectives[idx]["discovery_position"] = {
                "x": max(5, min(95, target_pos["x"] + discovery_offset_x)),
                "y": max(5, min(95, target_pos["y"] + discovery_offset_y))
            }
            objectives[idx]["discovery_radius"] = 15.0
            # Store hidden objectives separately
            self.hidden_objectives.append(objectives[idx])
        
        # Add time pressure for medium/hard difficulty
        if config["time_pressure"]:
            visible_objectives = [obj for obj in objectives if obj["status"] != "hidden"]
            for obj in visible_objectives[:2]:  # First 2 visible objectives are time-critical
                obj["time_limit"] = 120  # 2 minutes
        
        return objectives
    
    def _select_objective_type(self) -> ObjectiveType:
        """Select objective type based on difficulty and randomness"""
        types = list(ObjectiveType)
        # Ensure weights match the number of objective types (6 total now)
        if self.difficulty == "normal":
            weights = [3, 2, 1, 1, 2, 1]  # Favor move and recon
        elif self.difficulty == "medium":
            weights = [2, 2, 2, 2, 1, 1]  # Balanced, less resupply
        else:  # hard
            weights = [1, 3, 3, 3, 2, 1]  # Favor combat objectives
        return random.choices(types, weights=weights)[0]
    
    def _create_objective(self, obj_id: str, obj_type: ObjectiveType) -> Dict:
        """Create a specific mission objective"""
        target_pos = {
            "x": random.uniform(5, 95),
            "y": random.uniform(5, 95)
        }
        descriptions = {
            ObjectiveType.MOVE_TO_POSITION: f"Move team to position ({target_pos['x']:.1f}, {target_pos['y']:.1f})",
            ObjectiveType.SECURE_AREA: f"Secure and hold area at ({target_pos['x']:.1f}, {target_pos['y']:.1f})",
            ObjectiveType.EXTRACT_TARGET: f"Extract VIP from location ({target_pos['x']:.1f}, {target_pos['y']:.1f})",
            ObjectiveType.DEFEND_POSITION: f"Defend position ({target_pos['x']:.1f}, {target_pos['y']:.1f}) from enemy assault",
            ObjectiveType.RECONNAISSANCE: f"Conduct reconnaissance of area ({target_pos['x']:.1f}, {target_pos['y']:.1f})",
            ObjectiveType.RESUPPLY: f"Resupply team at supply depot ({target_pos['x']:.1f}, {target_pos['y']:.1f})"
        }
        score_values = {
            ObjectiveType.MOVE_TO_POSITION: 50,
            ObjectiveType.SECURE_AREA: 100,
            ObjectiveType.EXTRACT_TARGET: 150,
            ObjectiveType.DEFEND_POSITION: 120,
            ObjectiveType.RECONNAISSANCE: 80,
            ObjectiveType.RESUPPLY: 60
        }
        return {
            "obj_id": obj_id,
            "obj_type": obj_type.value,
            "description": descriptions[obj_type],
            "target_position": target_pos,
            "score_value": score_values[obj_type],
            "status": "pending",
            "assigned_members": [],
            "start_time": None,
            "time_limit": None
        }
    
    def check_objectives(self, mission_objectives: List[Dict], 
                        completed: List[Dict], 
                        failed: List[Dict],
                        team_status: Dict) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Check objective completion, failure conditions, and hidden objective discovery"""
        new_completed = []
        new_failed = []
        newly_discovered = []
        
        # Check for hidden objective discoveries
        newly_discovered = self._check_hidden_objectives_discovery(team_status)
        
        for obj in mission_objectives:
            if obj in completed or obj in failed:
                continue
            
            # Skip hidden objectives for completion/failure checks
            if obj.get("status") == "hidden":
                continue
            
            # Check completion conditions
            if self._check_objective_completion(obj, team_status):
                obj["status"] = "completed"
                new_completed.append(obj)
                continue
            
            # Check failure conditions
            if self._check_objective_failure(obj, team_status):
                obj["status"] = "failed"
                new_failed.append(obj)
                continue
        
        return new_completed, new_failed, newly_discovered
    
    def _check_hidden_objectives_discovery(self, team_status: Dict) -> List[Dict]:
        """Check if any hidden objectives have been discovered by team exploration"""
        newly_discovered = []
        team_members = team_status.get("members", {})
        
        for obj in self.hidden_objectives.copy():  # Use copy to avoid modification during iteration
            if obj.get("status") != "hidden":
                continue
                
            discovery_pos = obj.get("discovery_position")
            if not discovery_pos:
                continue
                
            discovery_radius = obj.get("discovery_radius", 15.0)
            
            # Check if any team member is within discovery radius
            for member_id, member_info in team_members.items():
                member_pos = member_info.get("position", {})
                if not member_pos:
                    continue
                    
                distance = ((member_pos["x"] - discovery_pos["x"]) ** 2 + 
                           (member_pos["y"] - discovery_pos["y"]) ** 2) ** 0.5
                
                if distance <= discovery_radius:
                    # Check if this member has scout role (scouts are better at discovery)
                    member_role = member_info.get("role", "")
                    
                    # Discovery probability based on role
                    discovery_chance = 0.8 if member_role == "scout" else 0.4
                    
                    if random.random() < discovery_chance:
                        # Objective discovered!
                        obj["status"] = "pending"
                        obj["discovered_by"] = member_id
                        newly_discovered.append(obj)
                        
                        # Remove from hidden objectives list
                        if obj in self.hidden_objectives:
                            self.hidden_objectives.remove(obj)
                        break
        
        return newly_discovered
    
    def get_visible_objectives(self, all_objectives: List[Dict]) -> List[Dict]:
        """Get only the objectives that are visible to the player (not hidden)"""
        return [obj for obj in all_objectives if obj.get("status") != "hidden"]
    
    def get_discovery_hints(self, team_status: Dict) -> List[str]:
        """Generate hints about areas that might contain hidden objectives"""
        hints = []
        team_members = team_status.get("members", {})
        
        for obj in self.hidden_objectives:
            discovery_pos = obj.get("discovery_position")
            if not discovery_pos:
                continue
                
            # Check if any team member is somewhat close (within 25 units) to discovery zone
            for member_id, member_info in team_members.items():
                member_pos = member_info.get("position", {})
                if not member_pos:
                    continue
                    
                distance = ((member_pos["x"] - discovery_pos["x"]) ** 2 + 
                           (member_pos["y"] - discovery_pos["y"]) ** 2) ** 0.5
                
                if 15 < distance <= 25:  # Close but not close enough to discover
                    hints.append(f"Team member {member_id} reports unusual activity in the area...")
                    break
        
        return hints
    
    def _check_objective_completion(self, objective: Dict, team_status: Dict) -> bool:
        """Check if objective is completed"""
        target_pos = objective.get("target_position")
        if not target_pos:
            return False
        
        team_members = team_status.get("members", {})
        
        # Check if any team member is at target position (within tolerance)
        tolerance = 5.0
        for member_id, member_info in team_members.items():
            member_pos = member_info.get("position", {})
            if member_pos:
                distance = ((member_pos["x"] - target_pos["x"]) ** 2 + 
                           (member_pos["y"] - target_pos["y"]) ** 2) ** 0.5
                
                if distance <= tolerance:
                    obj_type = objective.get("obj_type")
                    # Different objectives have different completion criteria
                    if obj_type == "move_to_position":
                        return True
                    elif obj_type == "reconnaissance":
                        # Scout role preferred for recon
                        if member_info.get("role") == "scout":
                            return True
                        # Any member can do recon if no scout available
                        return True
                    elif obj_type in ["secure_area", "defend_position"]:
                        # Need to stay at position for some time
                        return True  # Simplified for now
                    elif obj_type == "extract_target":
                        # Any team member can extract target
                        return True
                    elif obj_type == "resupply":
                        # Any team member can collect supplies
                        return True
        
        return False
    
    def _check_objective_failure(self, objective: Dict, team_status: Dict) -> bool:
        """Check if objective has failed"""
        current_time = time.time()
        
        # Time limit failure
        time_limit = objective.get("time_limit")
        start_time = objective.get("start_time")
        if time_limit and start_time:
            if current_time - start_time > time_limit:
                return True
        
        # Team incapacitation failure
        team_members = team_status.get("members", {})
        active_members = [m for m in team_members.values() 
                         if m.get("status") not in ["down", "injured"]]
        
        if len(active_members) == 0:
            return True
        
        return False
    
    def check_mission_complete(self, mission_objectives: List[Dict], 
                              completed_objectives: List[Dict]) -> bool:
        """Check if mission is complete"""
        # Mission complete if 80% of objectives completed
        completion_threshold = 0.8
        if len(mission_objectives) == 0:
            return True
        
        completion_ratio = len(completed_objectives) / len(mission_objectives)
        return completion_ratio >= completion_threshold
    
    def check_mission_failed(self, team_status: Dict, failed_objectives: List[Dict]) -> bool:
        """Check if mission has failed"""
        team_members = team_status.get("members", {})
        
        # Mission fails if all team members are down
        active_members = [m for m in team_members.values() 
                         if m.get("status") not in ["down"]]
        
        if len(active_members) == 0:
            return True
        
        # Mission fails if too many critical objectives failed
        critical_failures = len([obj for obj in failed_objectives 
                               if obj.get("obj_type") in ["extract_target", "defend_position"]])
        
        return critical_failures >= 2
    
    def calculate_final_score(self, completed_objectives: List[Dict],
                            failed_objectives: List[Dict],
                            rounds_used: float, max_rounds: float) -> int:
        """Calculate final mission score (updated for round-based gameplay)"""
        # Base score from completed objectives
        base_score = sum(obj.get("score_value", 0) for obj in completed_objectives)
        
        # Round efficiency bonus (fewer rounds used = higher score)
        if max_rounds > 0:
            rounds_ratio = rounds_used / max_rounds
            rounds_bonus = int(base_score * (1.0 - rounds_ratio) * 0.5)
        else:
            rounds_bonus = 0
        
        # Command efficiency bonus
        if self.total_commands_issued > 0:
            efficiency_ratio = self.successful_commands / self.total_commands_issued
            efficiency_bonus = int(base_score * efficiency_ratio * 0.3)
        else:
            efficiency_bonus = 0
        
        # Failure penalty
        failure_penalty = sum(obj.get("score_value", 0) for obj in failed_objectives) // 2
        
        final_score = max(0, base_score + rounds_bonus + efficiency_bonus - failure_penalty)
        
        return final_score
    
    def calculate_main_score(self, completed_objectives: List[Dict], 
                           failed_objectives: List[Dict], 
                           mission_objectives: List[Dict]) -> int:
        """Calculate current main score based on completed objectives"""
        # Base score from completed objectives
        base_score = sum(obj.get("score_value", 0) for obj in completed_objectives)
        
        # Small penalty for failed objectives
        failure_penalty = sum(obj.get("score_value", 0) for obj in failed_objectives) // 4
        
        return max(0, base_score - failure_penalty)
    
    def calculate_normalized_score(self, main_score: int, auxiliary_command_score: int,
                                 completed_objectives: List[Dict], mission_objectives: List[Dict],
                                 current_round: int, optimal_rounds: int) -> float:
        """Calculate normalized score (0-100 range) for real-time feedback"""
        # Calculate current total score
        total_score = main_score + auxiliary_command_score
        
        # Calculate theoretical maximum score for normalization
        if mission_objectives:
            max_objective_score = sum(obj.get("score_value", 0) for obj in mission_objectives)
        else:
            # Default estimates by difficulty
            base_scores = {'normal': 300, 'medium': 600, 'hard': 900}
            max_objective_score = base_scores.get(self.difficulty, 600)
        
        # Add estimated maximum auxiliary score
        estimated_max_auxiliary = max_objective_score // 2  # Rough estimate
        
        # Efficiency bonus estimate (if completed optimally)
        if optimal_rounds > 0 and current_round > 0:
            efficiency_factor = min(1.0, optimal_rounds / current_round)
            efficiency_bonus = int(max_objective_score * efficiency_factor * 0.3)
        else:
            efficiency_bonus = 0
        
        estimated_max_total = max_objective_score + estimated_max_auxiliary + efficiency_bonus
        
        if estimated_max_total > 0:
            normalized = min(100.0, max(0.0, (total_score / estimated_max_total) * 100))
        else:
            normalized = 0.0
        
        return normalized
    
    def update(self, dt: float, team_status: Dict):
        """Update game logic state"""
        command_history = team_status.get("command_history", [])
        self.total_commands_issued = len([cmd for cmd in command_history
                                        if cmd.get("timestamp", 0) > 0])
        
        self.successful_commands = len([cmd for cmd in command_history
                                      if cmd.get("success", False)])
        
        # Dynamic objective generation for hard difficulty
        if (self.difficulty_config.get("dynamic_objectives", False) and 
            random.random() < 0.001):  # Small chance each update
            self._add_dynamic_objective()
    
    def _add_dynamic_objective(self):
        """Add dynamic objective during mission (hard difficulty)"""
        new_obj = self._create_objective(
            f"dynamic_{int(time.time())}", 
            ObjectiveType.RECONNAISSANCE
        )
        new_obj["description"] = f"URGENT: {new_obj['description']}"
        new_obj["score_value"] = 200  # Higher value for dynamic objectives
        new_obj["time_limit"] = 90  # Short time limit
        
        self.current_objectives.append(new_obj) 