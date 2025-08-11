"""
Team Member System

Manages individual team members with their voice libraries, behavior patterns,
and status reporting capabilities.
"""

import random
import time
import os
import json
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass


class TeamMemberRole(Enum):
    """Team member roles defining their capabilities and behavior patterns"""
    SCOUT = "scout"
    HEAVY = "heavy"
    MEDIC = "medic"
    ENGINEER = "engineer"
    SNIPER = "sniper"
    SUPPORT = "support"


class TeamMemberStatus(Enum):
    """Current status of team members"""
    ACTIVE = "active"
    INJURED = "injured"
    BUSY = "busy"
    READY = "ready"
    DOWN = "down"


@dataclass
class Position:
    """2D position coordinates"""
    x: float
    y: float
    
    def distance_to(self, other: 'Position') -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


@dataclass
class VoiceAsset:
    """Voice asset for team member communication"""
    text: str
    audio_path: Optional[str] = None
    priority: int = 1  # 1=low, 5=urgent
    

class TeamMember:
    """Individual team member with voice library and behavior patterns"""
    
    def __init__(self, member_id: str, name: str, role: TeamMemberRole, 
                 position: Position, voice_library: Dict[str, List[VoiceAsset]],
                 deterministic_commands: bool = True):
        self.member_id = member_id
        self.name = name
        self.role = role
        self.position = position
        self.voice_library = voice_library
        self.deterministic_commands = deterministic_commands
        
        # Status tracking
        self.status = TeamMemberStatus.READY
        self.health = 100
        self.ammo = 100
        self.last_report_time = 0
        self.current_task = None
        self.task_progress = 0.0
        self.last_command_time = 0.0  # Track when last command was executed
        self.command_cooldown = 0.0   # No cooldown for turn-based gameplay
        
        # Behavior parameters
        self.report_frequency = self._get_report_frequency()
        self.success_rate = self._get_base_success_rate()
        self.movement_speed = self._get_movement_speed()
        self.movement_precision = self._get_movement_precision()  # Precision for movement commands
        
    def _get_report_frequency(self) -> float:
        """Get reporting frequency based on role"""
        frequencies = {
            TeamMemberRole.SCOUT: 15.0,  # Every 15 seconds
            TeamMemberRole.HEAVY: 30.0,
            TeamMemberRole.MEDIC: 25.0,
            TeamMemberRole.ENGINEER: 20.0,
            TeamMemberRole.SNIPER: 40.0,
            TeamMemberRole.SUPPORT: 30.0
        }
        return frequencies.get(self.role, 30.0)
    
    def _get_base_success_rate(self) -> float:
        """Get base success rate for tasks"""
        rates = {
            TeamMemberRole.SCOUT: 0.85,
            TeamMemberRole.HEAVY: 0.75,
            TeamMemberRole.MEDIC: 0.90,
            TeamMemberRole.ENGINEER: 0.80,
            TeamMemberRole.SNIPER: 0.70,
            TeamMemberRole.SUPPORT: 0.80
        }
        return rates.get(self.role, 0.75)
    
    def _get_movement_speed(self) -> float:
        """Get movement speed based on role"""
        speeds = {
            TeamMemberRole.SCOUT: 1.5,
            TeamMemberRole.HEAVY: 0.7,
            TeamMemberRole.MEDIC: 1.0,
            TeamMemberRole.ENGINEER: 0.9,
            TeamMemberRole.SNIPER: 1.1,
            TeamMemberRole.SUPPORT: 1.0
        }
        return speeds.get(self.role, 1.0)
    
    def _get_movement_precision(self) -> float:
        """Get movement precision (lower values = more accurate)"""
        precisions = {
            TeamMemberRole.SCOUT: 0.5,    # Most accurate
            TeamMemberRole.SNIPER: 0.7,   # Very accurate
            TeamMemberRole.MEDIC: 1.0,    # Accurate
            TeamMemberRole.ENGINEER: 1.2, # Somewhat accurate
            TeamMemberRole.SUPPORT: 1.5,  # Less accurate
            TeamMemberRole.HEAVY: 2.0,    # Least accurate
        }
        return precisions.get(self.role, 1.0)
    
    def should_report(self, current_time: float) -> bool:
        """Check if member should make a status report"""
        return (current_time - self.last_report_time) >= self.report_frequency
    
    def get_status_report(self) -> VoiceAsset:
        """Generate a status report message"""
        report_type = self._determine_report_type()
        
        if report_type not in self.voice_library:
            # Fallback to generic status report
            report_type = "status"
            
        if report_type in self.voice_library:
            report = random.choice(self.voice_library[report_type])
            return VoiceAsset(
                text=report.text.format(
                    name=self.name,
                    health=self.health,
                    ammo=self.ammo,
                    position_x=int(self.position.x),
                    position_y=int(self.position.y)
                ),
                audio_path=report.audio_path,
                priority=report.priority
            )
        
        # Ultimate fallback
        return VoiceAsset(
            text=f"{self.name}: All clear, standing by.",
            priority=1
        )
    
    def _determine_report_type(self) -> str:
        """Determine what type of report to make based on current status"""
        if self.health < 30:
            return "injured"
        elif self.ammo < 20:
            return "low_ammo"
        elif self.status == TeamMemberStatus.BUSY:
            return "commands"  # Use commands category for task progress
        elif random.random() < 0.3:  # 30% chance of environmental report
            return "environment"
        else:
            return "status"
    
    def execute_command(self, command: str, target_position: Optional[Position] = None) -> Tuple[bool, str]:
        """Execute a command from the commander"""
        current_time = time.time()
        
        # Check if member is available to receive commands
        if self.status == TeamMemberStatus.BUSY:
            return False, f"{self.name}: Currently busy with another task, cannot accept new orders."
        
        if self.status == TeamMemberStatus.DOWN:
            return False, f"{self.name}: Unit down, not responding to commands."
        
        if self.health <= 10:  # Critical health threshold
            return False, f"{self.name}: Critical health status, unable to execute commands."
        
        # Check command cooldown to prevent spam (disabled for turn-based gameplay)
        if self.command_cooldown > 0 and current_time - self.last_command_time < self.command_cooldown:
            remaining_cooldown = self.command_cooldown - (current_time - self.last_command_time)
            return False, f"{self.name}: Still executing previous command, wait {remaining_cooldown:.1f}s."
        
        # Calculate success based on command clarity and member capability
        if self.deterministic_commands:
            # Deterministic mode: all valid commands succeed
            success = True
        else:
            # Probabilistic mode: calculate success based on member capability
            base_success = self.success_rate
            
            # Adjust success rate based on health and status
            if self.health < 50:
                base_success *= 0.7
            if self.status == TeamMemberStatus.INJURED:
                base_success *= 0.5
                
            success = random.random() < base_success
        
        if success:
            # Update last command time and set status
            self.last_command_time = current_time
            if self.status == TeamMemberStatus.READY:
                self.status = TeamMemberStatus.ACTIVE
            
            if target_position:
                # Apply movement uncertainty based on role precision and environmental factors
                actual_position = self._calculate_actual_movement_position(target_position)
                
                # Clamp position to map boundaries (0-100 for both x and y)
                clamped_x = max(0, min(100, actual_position.x))
                clamped_y = max(0, min(100, actual_position.y))
                self.position = Position(clamped_x, clamped_y)
                
                # Calculate movement error for feedback
                intended_x, intended_y = target_position.x, target_position.y
                error_distance = ((clamped_x - intended_x) ** 2 + (clamped_y - intended_y) ** 2) ** 0.5
                
                if error_distance > 2.0:  # Significant error
                    return True, f"{self.name}: Moving to position {clamped_x:.1f},{clamped_y:.1f} (slight deviation from target {intended_x:.1f},{intended_y:.1f})."
                elif clamped_x != target_position.x or clamped_y != target_position.y:
                    return True, f"{self.name}: Moving to position {clamped_x:.1f},{clamped_y:.1f} (adjusted to stay within map bounds)."
                else:
                    return True, f"{self.name}: Moving to position {clamped_x:.1f},{clamped_y:.1f}."
                    
            return True, f"{self.name}: Command acknowledged and executing."
        else:
            failure_reasons = [
                f"{self.name}: Unable to comply, path blocked.",
                f"{self.name}: Negative, taking enemy fire.",
                f"{self.name}: Cannot complete, equipment malfunction.",
                f"{self.name}: Request clarification on orders."
            ]
            return False, random.choice(failure_reasons)
    
    def _calculate_actual_movement_position(self, target_position: Position) -> Position:
        """Calculate actual movement position with uncertainty based on movement distance"""
        # Calculate movement distance from current position to target
        movement_distance = self.position.distance_to(target_position)
        
        # Base error coefficient from role precision
        # Scout: 0.5 -> 0.025, Heavy: 2.0 -> 0.1, etc.
        base_error_coefficient = self.movement_precision * 0.05
        
        # Environmental factors affecting precision
        environmental_multiplier = 1.0
        
        # Health affects precision
        if self.health < 50:
            environmental_multiplier += 0.5
        if self.health < 25:
            environmental_multiplier += 0.5
            
        # Status affects precision
        if self.status == TeamMemberStatus.INJURED:
            environmental_multiplier += 0.75
        
        # Calculate total error coefficient
        total_error_coefficient = base_error_coefficient * environmental_multiplier
        
        # Calculate movement error: distance * coefficient (just like your example)
        movement_error = movement_distance * total_error_coefficient
        
        # Distribute error randomly to x and y coordinates
        # Each coordinate gets a random portion of the total error
        error_x = random.uniform(-movement_error/2, movement_error/2)
        error_y = random.uniform(-movement_error/2, movement_error/2)
        
        # Calculate actual position
        actual_x = target_position.x + error_x
        actual_y = target_position.y + error_y
        
        return Position(actual_x, actual_y)
    
    def update(self, dt: float):
        """Update team member state"""
        # Simulate gradual changes
        if self.status == TeamMemberStatus.INJURED and random.random() < 0.02:
            self.health = max(10, self.health - 1)
        
        # Task progress
        if self.current_task and self.status == TeamMemberStatus.BUSY:
            self.task_progress += dt * 0.1  # 10 seconds to complete task
            if self.task_progress >= 1.0:
                self.current_task = None
                self.task_progress = 0.0
                self.status = TeamMemberStatus.READY


class TeamMemberManager:
    """Manages all team members and their interactions"""
    
    def __init__(self, difficulty: str = "normal", deterministic_commands: bool = True):
        self.difficulty = difficulty
        self.deterministic_commands = deterministic_commands
        self.members: Dict[str, TeamMember] = {}
        self.voice_libraries = {}
        self.current_time = 0.0
        
        # Load voice libraries and create team members
        self._load_voice_libraries()
        self._create_team_members()
    
    def _load_voice_libraries(self):
        """Load voice libraries for different team member types"""
        # Simplified voice libraries without audio files for demo
        
        # Scout voice library
        scout_voices = {
            "status": [
                VoiceAsset("Scout {name}: Position {position_x},{position_y}, all clear.", None, 1),
                VoiceAsset("Scout {name}: Area secured, no contacts.", None, 1),
                VoiceAsset("Scout {name}: Maintaining overwatch position.", None, 1),
            ],
            "environment": [
                VoiceAsset("Scout {name}: Movement detected in sector 7.", None, 3),
                VoiceAsset("Scout {name}: Possible enemy patrol spotted.", None, 4),
                VoiceAsset("Scout {name}: High ground secured, good visibility.", None, 2),
                VoiceAsset("Scout {name}: Enemy contact, engaging!", None, 4),
                VoiceAsset("Scout {name}: Target acquired, requesting fire support.", None, 4),
            ],
            "injured": [
                VoiceAsset("Scout {name}: Taking damage, need support!", None, 5),
                VoiceAsset("Scout {name}: Hit! Health critical!", None, 5),
            ],
            "low_ammo": [
                VoiceAsset("Scout {name}: Running low on ammunition.", None, 3),
                VoiceAsset("Scout {name}: Ammo depleted, need resupply.", None, 3),
            ],
            "commands": [
                VoiceAsset("Scout {name}: Command acknowledged and executing.", None, 2),
                VoiceAsset("Scout {name}: Unable to comply, path blocked.", None, 2),
                VoiceAsset("Scout {name}: Negative, taking enemy fire.", None, 3),
                VoiceAsset("Scout {name}: Request clarification on orders.", None, 2),
            ]
        }
        
        # Heavy weapons specialist voice library
        heavy_voices = {
            "status": [
                VoiceAsset("Heavy {name}: In position, ready for action.", None, 1),
                VoiceAsset("Heavy {name}: Standing by with heavy weapons.", None, 1),
                VoiceAsset("Heavy {name}: Sector clear, moving to next position.", None, 1),
            ],
            "environment": [
                VoiceAsset("Heavy {name}: Enemy fortification spotted.", None, 4),
                VoiceAsset("Heavy {name}: Multiple contacts, engaging!", None, 5),
                VoiceAsset("Heavy {name}: Area suppressed, advance possible.", None, 3),
                VoiceAsset("Heavy {name}: Target neutralized.", None, 2),
            ],
            "injured": [
                VoiceAsset("Heavy {name}: Taking heavy fire! Need medic!", None, 5),
                VoiceAsset("Heavy {name}: Armor damaged, health critical!", None, 5),
            ],
            "low_ammo": [
                VoiceAsset("Heavy {name}: Heavy weapons ammo depleted.", None, 4),
                VoiceAsset("Heavy {name}: Need immediate resupply!", None, 4),
            ],
            "commands": [
                VoiceAsset("Heavy {name}: Command acknowledged and executing.", None, 2),
                VoiceAsset("Heavy {name}: Unable to comply, path blocked.", None, 2),
                VoiceAsset("Heavy {name}: Negative, taking enemy fire.", None, 3),
            ]
        }
        
        # Medic voice library
        medic_voices = {
            "status": [
                VoiceAsset("Medic {name}: Medical station operational.", None, 1),
                VoiceAsset("Medic {name}: All team members in good health.", None, 1),
                VoiceAsset("Medic {name}: Medical supplies adequate.", None, 1),
            ],
            "environment": [
                VoiceAsset("Medic {name}: Casualties in the area.", None, 3),
                VoiceAsset("Medic {name}: Safe zone established for treatment.", None, 2),
                VoiceAsset("Medic {name}: Medical evacuation required.", None, 4),
            ],
            "injured": [
                VoiceAsset("Medic {name}: Medic down! Need backup!", None, 5),
                VoiceAsset("Medic {name}: Cannot treat others, need help!", None, 5),
            ],
            "low_ammo": [
                VoiceAsset("Medic {name}: Low on medical supplies.", None, 3),
            ],
            "commands": [
                VoiceAsset("Medic {name}: Command acknowledged and executing.", None, 2),
                VoiceAsset("Medic {name}: Moving to wounded teammate.", None, 3),
            ]
        }
        
        # Engineer voice library
        engineer_voices = {
            "status": [
                VoiceAsset("Engineer {name}: Engineering station online.", None, 1),
                VoiceAsset("Engineer {name}: Equipment maintenance complete.", None, 1),
                VoiceAsset("Engineer {name}: Defensive positions ready.", None, 1),
            ],
            "environment": [
                VoiceAsset("Engineer {name}: Explosive devices armed.", None, 3),
                VoiceAsset("Engineer {name}: Deploying defensive barriers.", None, 2),
                VoiceAsset("Engineer {name}: Setting up automated turrets.", None, 3),
            ],
            "injured": [
                VoiceAsset("Engineer {name}: Taking damage, need support!", None, 5),
                VoiceAsset("Engineer {name}: Equipment damaged, need repairs!", None, 5),
            ],
            "low_ammo": [
                VoiceAsset("Engineer {name}: Low on engineering supplies.", None, 3),
                VoiceAsset("Engineer {name}: Need immediate resupply!", None, 4),
            ],
            "commands": [
                VoiceAsset("Engineer {name}: Command acknowledged and executing.", None, 2),
                VoiceAsset("Engineer {name}: Repairing damaged equipment.", None, 2),
                VoiceAsset("Engineer {name}: Breaching obstacles.", None, 3),
            ]
        }
        
        # Sniper voice library
        sniper_voices = {
            "status": [
                VoiceAsset("Sniper {name}: Overwatch position established.", None, 1),
                VoiceAsset("Sniper {name}: Target in crosshairs.", None, 1),
                VoiceAsset("Sniper {name}: Scanning for threats.", None, 1),
            ],
            "environment": [
                VoiceAsset("Sniper {name}: Multiple targets identified.", None, 3),
                VoiceAsset("Sniper {name}: High value target spotted.", None, 4),
                VoiceAsset("Sniper {name}: Target eliminated.", None, 2),
            ],
            "injured": [
                VoiceAsset("Sniper {name}: Taking fire, relocating!", None, 5),
                VoiceAsset("Sniper {name}: Position compromised, need extraction!", None, 5),
            ],
            "low_ammo": [
                VoiceAsset("Sniper {name}: Low on specialized ammunition.", None, 3),
                VoiceAsset("Sniper {name}: Need precision ammo resupply.", None, 3),
            ],
            "commands": [
                VoiceAsset("Sniper {name}: Command acknowledged and executing.", None, 2),
                VoiceAsset("Sniper {name}: Providing overwatch support.", None, 2),
                VoiceAsset("Sniper {name}: Moving to elevated position.", None, 2),
            ]
        }
        
        # Support specialist voice library
        support_voices = {
            "status": [
                VoiceAsset("Support {name}: Support systems online.", None, 1),
                VoiceAsset("Support {name}: Communications network active.", None, 1),
                VoiceAsset("Support {name}: Supply lines secure.", None, 1),
            ],
            "environment": [
                VoiceAsset("Support {name}: Tactical data updated.", None, 2),
                VoiceAsset("Support {name}: Intelligence gathering complete.", None, 2),
                VoiceAsset("Support {name}: Coordinating team movements.", None, 2),
            ],
            "injured": [
                VoiceAsset("Support {name}: Support unit under attack!", None, 5),
                VoiceAsset("Support {name}: Communications disrupted!", None, 5),
            ],
            "low_ammo": [
                VoiceAsset("Support {name}: Support equipment damaged.", None, 3),
                VoiceAsset("Support {name}: Need equipment resupply.", None, 3),
            ],
            "commands": [
                VoiceAsset("Support {name}: Command acknowledged and executing.", None, 2),
                VoiceAsset("Support {name}: Relaying tactical information.", None, 2),
                VoiceAsset("Support {name}: Providing logistical support.", None, 2),
            ]
        }
        
        # Store voice libraries by role
        self.voice_libraries = {
            TeamMemberRole.SCOUT: scout_voices,
            TeamMemberRole.HEAVY: heavy_voices,
            TeamMemberRole.MEDIC: medic_voices,
            TeamMemberRole.ENGINEER: engineer_voices,
            TeamMemberRole.SNIPER: sniper_voices,
            TeamMemberRole.SUPPORT: support_voices
        }
    
    def _create_team_members(self):
        """Create team members based on difficulty level"""
        team_configs = {
            "normal": [
                ("scout_1", "Alpha", TeamMemberRole.SCOUT, Position(10, 10)),
                ("heavy_1", "Bravo", TeamMemberRole.HEAVY, Position(15, 8)),
            ],
            "medium": [
                ("scout_1", "Alpha", TeamMemberRole.SCOUT, Position(10, 10)),
                ("heavy_1", "Bravo", TeamMemberRole.HEAVY, Position(15, 8)),
                ("medic_1", "Charlie", TeamMemberRole.MEDIC, Position(12, 12)),
                ("engineer_1", "Delta", TeamMemberRole.ENGINEER, Position(8, 15)),
            ],
            "hard": [
                ("scout_1", "Alpha", TeamMemberRole.SCOUT, Position(10, 10)),
                ("heavy_1", "Bravo", TeamMemberRole.HEAVY, Position(15, 8)),
                ("medic_1", "Charlie", TeamMemberRole.MEDIC, Position(12, 12)),
                ("engineer_1", "Delta", TeamMemberRole.ENGINEER, Position(8, 15)),
                ("sniper_1", "Echo", TeamMemberRole.SNIPER, Position(20, 20)),
                ("support_1", "Foxtrot", TeamMemberRole.SUPPORT, Position(5, 5)),
            ]
        }
        
        config = team_configs.get(self.difficulty, team_configs["normal"])
        
        for member_id, name, role, position in config:
            voice_lib = self.voice_libraries.get(role, self.voice_libraries[TeamMemberRole.SCOUT])
            member = TeamMember(member_id, name, role, position, voice_lib, self.deterministic_commands)
            self.members[member_id] = member
    
    def get_pending_reports(self) -> List[Tuple[str, VoiceAsset]]:
        """Get all pending status reports from team members"""
        reports = []
        for member_id, member in self.members.items():
            if member.should_report(self.current_time):
                report = member.get_status_report()
                reports.append((member_id, report))
                member.last_report_time = self.current_time
        return reports
    
    def execute_command(self, member_id: str, command: str, 
                       target_position = None) -> Tuple[bool, str]:
        """Execute command for specific team member"""
        if member_id not in self.members:
            return False, f"Team member {member_id} not found in roster."
        
        member = self.members[member_id]
        
        # Check if member exists but is in an invalid state
        if not member:
            return False, f"Team member {member_id} is in an invalid state."
        
        # Convert tuple to Position object if needed
        position_obj = None
        if target_position is not None:
            if isinstance(target_position, tuple) and len(target_position) == 2:
                position_obj = Position(float(target_position[0]), float(target_position[1]))
            elif isinstance(target_position, Position):
                position_obj = target_position
            else:
                return False, f"Invalid target_position format: {type(target_position)}"
        
        try:
            return member.execute_command(command, position_obj)
        except Exception as e:
            member_name = getattr(member, 'name', member_id)
            return False, f"Command execution failed for {member_name}: {str(e)}"
    
    def get_team_status(self) -> Dict:
        """Get comprehensive team status"""
        status = {
            "total_members": len(self.members),
            "active_members": sum(1 for m in self.members.values() if m.status == TeamMemberStatus.ACTIVE),
            "injured_members": sum(1 for m in self.members.values() if m.status == TeamMemberStatus.INJURED),
            "members": {}
        }
        
        for member_id, member in self.members.items():
            # Ensure all member data is in consistent dictionary format
            member_data = {
                "name": member.name,
                "role": member.role.value,
                "status": member.status.value,
                "health": float(member.health),
                "ammo": float(member.ammo),
                "position": {"x": float(member.position.x), "y": float(member.position.y)},
                "current_task": member.current_task,
                "task_progress": float(member.task_progress)
            }
            
            # Validate the format to catch any issues early
            self._validate_member_data(member_data, member_id)
            status["members"][member_id] = member_data
        
        return status
    
    def _validate_member_data(self, member_data: Dict, member_id: str):
        """Validate that member data has the correct format and types."""
        required_fields = ["name", "role", "status", "health", "ammo", "position", "current_task", "task_progress"]
        
        for field in required_fields:
            if field not in member_data:
                raise ValueError(f"Member {member_id} missing required field: {field}")
        
        # Validate position structure
        if not isinstance(member_data["position"], dict):
            raise ValueError(f"Member {member_id} position must be a dict, got {type(member_data['position'])}")
        
        if "x" not in member_data["position"] or "y" not in member_data["position"]:
            raise ValueError(f"Member {member_id} position missing x or y coordinates")
        
        # Validate numeric fields
        try:
            float(member_data["health"])
            float(member_data["ammo"])
            float(member_data["position"]["x"])
            float(member_data["position"]["y"])
            float(member_data["task_progress"])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Member {member_id} has invalid numeric data: {e}")
    
    def get_member(self, member_id: str) -> Optional[TeamMember]:
        """Get specific team member by ID"""
        return self.members.get(member_id)
    
    def reset_team(self):
        """Reset all team members to initial state"""
        self.current_time = 0.0
        for member in self.members.values():
            member.health = 100
            member.ammo = 100
            member.status = TeamMemberStatus.READY
            member.current_task = None
            member.task_progress = 0.0
            member.last_report_time = 0.0
    
    def update(self, dt: float):
        """Update all team members"""
        self.current_time += dt
        for member in self.members.values():
            member.update(dt) 