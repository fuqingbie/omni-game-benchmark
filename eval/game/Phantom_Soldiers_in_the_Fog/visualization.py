"""
Game Visualization System

Provides visualization data for the UI, including map rendering, team member positions,
objective markers, and status displays.

Enhanced with semantic asset loading and caching capabilities.
"""

import math
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Import asset manager
try:
    from asset_manager import AssetManager
    ASSET_MANAGER_AVAILABLE = True
except ImportError:
    ASSET_MANAGER_AVAILABLE = False
    print("⚠️  Asset manager not available. Using basic visualization.")


class MapTileType(Enum):
    """Types of map tiles"""
    OPEN = "open"
    COVER = "cover"
    BUILDING = "building"
    WATER = "water"
    OBJECTIVE = "objective"


@dataclass
class MapTile:
    """Individual map tile"""
    x: int
    y: int
    tile_type: MapTileType
    passable: bool = True
    cover_value: float = 0.0  # 0.0 = no cover, 1.0 = full cover


class GameVisualization:
    """Handles game visualization and UI data preparation with only color support"""
    def __init__(self, map_width: int = 100, map_height: int = 100):
        self.map_width = map_width
        self.map_height = map_height
        self.tile_colors = {
            MapTileType.OPEN: "#90EE90",
            MapTileType.COVER: "#8B4513",
            MapTileType.BUILDING: "#696969",
            MapTileType.WATER: "#4169E1",
            MapTileType.OBJECTIVE: "#FFD700"
        }
        self.member_colors = {
            "scout": "#00FF00",
            "heavy": "#FF0000",
            "medic": "#FFFFFF",
            "engineer": "#FFFF00",
            "sniper": "#800080",
            "support": "#FFA500"
        }
        self.zoom_level = 1.0
        self.view_center = (50, 50)
        self.view_radius = 25
        self.map_tiles = self._generate_map()

    def _generate_map(self) -> Dict[Tuple[int, int], MapTile]:
        import random
        tiles = {}
        for x in range(self.map_width):
            for y in range(self.map_height):
                tiles[(x, y)] = MapTile(x, y, MapTileType.OPEN)
        # Add a few buildings and cover for demo
        for _ in range(10):
            bx, by = random.randint(10, self.map_width-10), random.randint(10, self.map_height-10)
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    x, y = bx+dx, by+dy
                    if 0 <= x < self.map_width and 0 <= y < self.map_height:
                        tiles[(x, y)] = MapTile(x, y, MapTileType.BUILDING, passable=False)
        for _ in range(20):
            x, y = random.randint(0, self.map_width-1), random.randint(0, self.map_height-1)
            if tiles[(x, y)].tile_type == MapTileType.OPEN:
                tiles[(x, y)] = MapTile(x, y, MapTileType.COVER, cover_value=0.5)
        return tiles
    
    def get_render_data(self, team_status: Dict, hidden_objectives: List[Dict] = None, visible_objectives: List[Dict] = None) -> Dict[str, Any]:
        """Get all data needed for rendering the game visualization"""
        # Get visible map area
        visible_tiles = self._get_visible_tiles()
        
        # Get team member positions and info
        team_members = self._format_team_members(team_status.get("members", {}))
        
        # Get objective markers - use provided visible_objectives or look in team_status for backwards compatibility
        objectives_to_render = visible_objectives if visible_objectives is not None else team_status.get("objectives", [])
        objective_markers = self._get_objective_markers(objectives_to_render, hidden_objectives)
        
        # Get tactical overlay information
        tactical_info = self._get_tactical_overlay(team_status)
        
        return {
            "map_data": {
                "tiles": visible_tiles,
                "width": self.map_width,
                "height": self.map_height,
                "view_center": self.view_center,
                "zoom_level": self.zoom_level,
                "assets_enabled": False
            },
            "team_members": team_members,
            "objectives": objective_markers,
            "tactical_overlay": tactical_info,
            "ui_elements": self._get_ui_elements(team_status),
            "asset_info": self._get_asset_info()
        }
    
    def _get_visible_tiles(self) -> List[Dict]:
        """Get tiles visible in current view"""
        visible_tiles = []
        
        cx, cy = self.view_center
        radius = int(self.view_radius / self.zoom_level)
        
        for x in range(max(0, cx - radius), min(self.map_width, cx + radius + 1)):
            for y in range(max(0, cy - radius), min(self.map_height, cy + radius + 1)):
                tile = self.map_tiles.get((x, y))
                if tile:
                    tile_data = {
                        "x": x,
                        "y": y,
                        "type": tile.tile_type.value,
                        "color": self.tile_colors[tile.tile_type],
                        "passable": tile.passable,
                        "cover_value": tile.cover_value
                    }
                    
                    visible_tiles.append(tile_data)
        
        return visible_tiles
    
    def _format_team_members(self, members: Dict) -> List[Dict]:
        """Format team member data for visualization"""
        formatted_members = []
        
        for member_id, member_info in members.items():
            position = member_info.get("position", {})
            if position:
                x = position.get("x", 0)
                y = position.get("y", 0)
                
                # Only include team members within map boundaries (small buffer for valid edge cases)
                if 0 <= x <= self.map_width and 0 <= y <= self.map_height:
                    role = member_info.get("role", "scout")
                    
                    member_data = {
                        "id": member_id,
                        "name": member_info.get("name", member_id),
                        "role": role,
                        "position": {
                            "x": x,
                            "y": y
                        },
                        "status": member_info.get("status", "unknown"),
                        "health": member_info.get("health", 100),
                        "ammo": member_info.get("ammo", 100),
                        "color": self.member_colors.get(role, "#FFFFFF"),
                        "current_task": member_info.get("current_task"),
                        "task_progress": member_info.get("task_progress", 0.0)
                    }
                    
                    formatted_members.append(member_data)
                else:
                    # Log warning for out-of-bounds team members
                    print(f"⚠️  Team member {member_id} at position ({x}, {y}) is outside map bounds ({self.map_width}x{self.map_height})")
        
        return formatted_members

    def _get_asset_info(self) -> Dict[str, Any]:
        """Get information about loaded assets"""
        return {"enabled": False}
    
    def _get_objective_markers(self, objectives: List[Dict], hidden_objectives: List[Dict] = None) -> List[Dict]:
        """Get objective markers for map display"""
        markers = []
        
        # Process visible objectives
        for obj in objectives:
            target_pos = obj.get("target_position", {})
            if target_pos:
                status = obj.get("status", "pending")
                
                # Skip hidden objectives in visible list
                if status == "hidden":
                    continue
                
                # Different colors for different objective states
                colors = {
                    "pending": "#FFFF00",    # Yellow
                    "in_progress": "#FFA500", # Orange
                    "completed": "#00FF00",   # Green
                    "failed": "#FF0000"       # Red
                }
                
                marker = {
                    "id": obj.get("obj_id", "unknown"),
                    "type": obj.get("obj_type", "unknown"),
                    "description": obj.get("description", "Unknown objective"),
                    "position": {
                        "x": target_pos.get("x", 0),
                        "y": target_pos.get("y", 0)
                    },
                    "status": status,
                    "color": colors.get(status, "#FFFFFF"),
                    "score_value": obj.get("score_value", 0),
                    "is_hidden": False
                }
                
                markers.append(marker)
        
        # Add discovery zones for hidden objectives (only show hints, not exact locations)
        if hidden_objectives:
            for obj in hidden_objectives:
                discovery_pos = obj.get("discovery_position", {})
                if discovery_pos:
                    # Only show discovery zone as a general area hint
                    marker = {
                        "id": f"discovery_{obj.get('obj_id', 'unknown')}",
                        "type": "discovery_zone",
                        "description": "Unexplored area - might contain objectives",
                        "position": {
                            "x": discovery_pos.get("x", 0),
                            "y": discovery_pos.get("y", 0)
                        },
                        "status": "hidden",
                        "color": "#808080",  # Gray for mystery
                        "radius": obj.get("discovery_radius", 15.0),
                        "is_hidden": True
                    }
                    
                    markers.append(marker)
        
        return markers
    
    def _get_tactical_overlay(self, team_status: Dict) -> Dict[str, Any]:
        """Get tactical overlay information (lines of sight, movement paths, etc.)"""
        overlay = {
            "movement_paths": [],
            "sight_lines": [],
            "danger_zones": [],
            "communication_lines": []
        }
        
        members = team_status.get("members", {})
        
        # Generate sight lines for each team member
        for member_id, member_info in members.items():
            position = member_info.get("position", {})
            role = member_info.get("role", "scout")
            
            if position:
                sight_range = self._get_sight_range(role)
                sight_lines = self._calculate_sight_lines(position, sight_range)
                
                overlay["sight_lines"].append({
                    "member_id": member_id,
                    "lines": sight_lines,
                    "range": sight_range
                })
        
        # Add communication lines between nearby team members
        member_positions = [(mid, minfo.get("position", {})) 
                           for mid, minfo in members.items()]
        
        for i, (m1_id, m1_pos) in enumerate(member_positions):
            for j, (m2_id, m2_pos) in enumerate(member_positions[i+1:], i+1):
                if m1_pos and m2_pos:
                    distance = math.sqrt((m1_pos["x"] - m2_pos["x"])**2 + 
                                       (m1_pos["y"] - m2_pos["y"])**2)
                    
                    # Show communication line if within reasonable distance
                    if distance <= 30:  # Communication range
                        overlay["communication_lines"].append({
                            "from": m1_id,
                            "to": m2_id,
                            "from_pos": m1_pos,
                            "to_pos": m2_pos,
                            "strength": max(0.1, 1.0 - distance / 30)
                        })
        
        return overlay
    
    def _get_sight_range(self, role: str) -> float:
        """Get sight range for different roles"""
        ranges = {
            "scout": 25.0,
            "sniper": 35.0,
            "heavy": 15.0,
            "medic": 18.0,
            "engineer": 20.0,
            "support": 22.0
        }
        return ranges.get(role, 20.0)
    
    def _calculate_sight_lines(self, position: Dict, sight_range: float) -> List[Dict]:
        """Calculate sight lines from position (simplified version)"""
        lines = []
        
        # Create sight lines in 8 directions
        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            
            # Cast ray to maximum range or until obstruction
            for distance in range(1, int(sight_range) + 1):
                x = position["x"] + distance * math.cos(rad)
                y = position["y"] + distance * math.sin(rad)
                
                # Check for obstacles
                tile_x, tile_y = int(x), int(y)
                if (tile_x, tile_y) in self.map_tiles:
                    tile = self.map_tiles[(tile_x, tile_y)]
                    if not tile.passable:
                        break
                
                # Add point to sight line
                if distance == int(sight_range) or distance % 5 == 0:
                    lines.append({
                        "x": x,
                        "y": y,
                        "distance": distance,
                        "angle": angle
                    })
        
        return lines
    
    def _get_ui_elements(self, team_status: Dict) -> Dict[str, Any]:
        """Get UI elements like minimap, status bars, etc."""
        return {
            "minimap": {
                "enabled": True,
                "scale": 0.2,
                "position": {"x": 10, "y": 10}
            },
            "status_bars": self._get_status_bars(team_status),
            "command_prompt": {
                "visible": True,
                "position": {"x": 10, "y": 400},
                "history": team_status.get("recent_commands", [])
            },
            "objective_list": {
                "visible": True,
                "position": {"x": 600, "y": 10},
                "objectives": team_status.get("mission_objectives", [])
            }
        }
    
    def _get_status_bars(self, team_status: Dict) -> List[Dict]:
        """Get status bar information for each team member"""
        status_bars = []
        
        members = team_status.get("members", {})
        for i, (member_id, member_info) in enumerate(members.items()):
            status_bars.append({
                "member_id": member_id,
                "name": member_info.get("name", member_id),
                "role": member_info.get("role", "unknown"),
                "health": member_info.get("health", 100),
                "ammo": member_info.get("ammo", 100),
                "status": member_info.get("status", "ready"),
                "position": {"x": 10, "y": 50 + i * 30},
                "color": self.member_colors.get(member_info.get("role", "scout"), "#FFFFFF")
            })
        
        return status_bars
    
    def update_view(self, center_x: float, center_y: float, zoom: float):
        """Update view center and zoom level"""
        # Ensure view center stays within map bounds with proper margins
        min_x = max(0, self.view_radius)
        max_x = min(self.map_width, self.map_width - self.view_radius)
        min_y = max(0, self.view_radius)
        max_y = min(self.map_height, self.map_height - self.view_radius)
        
        self.view_center = (
            max(min_x, min(max_x, center_x)),
            max(min_y, min(max_y, center_y))
        )
        self.zoom_level = max(0.5, min(3.0, zoom))
    
    def world_to_screen(self, world_x: float, world_y: float, 
                       screen_width: int, screen_height: int) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        cx, cy = self.view_center
        
        # Relative to view center
        rel_x = world_x - cx
        rel_y = world_y - cy
        
        # Apply zoom
        rel_x *= self.zoom_level
        rel_y *= self.zoom_level
        
        # Convert to screen coordinates
        screen_x = int(screen_width // 2 + rel_x * 10)  # 10 pixels per tile
        screen_y = int(screen_height // 2 + rel_y * 10)
        
        return screen_x, screen_y
    
    def screen_to_world(self, screen_x: int, screen_y: int, 
                       screen_width: int, screen_height: int) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates"""
        cx, cy = self.view_center
        
        # Convert from screen coordinates
        rel_x = (screen_x - screen_width // 2) / 10.0  # 10 pixels per tile
        rel_y = (screen_y - screen_height // 2) / 10.0
        
        # Apply inverse zoom
        rel_x /= self.zoom_level
        rel_y /= self.zoom_level
        
        # Convert to world coordinates
        world_x = cx + rel_x
        world_y = cy + rel_y
        
        return world_x, world_y 