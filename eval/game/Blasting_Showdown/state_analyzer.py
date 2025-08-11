import json
import os
import glob
import re
from collections import defaultdict
from datetime import datetime

class BombermanStatsAnalyzer:
    def __init__(self):
        self.model_stats = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'games': 0, 'kills': 0, 'deaths': 0, 'items': 0}))
        self.player_id_stats = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'games': 0, 'kills': 0, 'deaths': 0, 'items': 0, 'models': defaultdict(int)}))
        
    def extract_difficulty_from_filename(self, filename):
        """Extract difficulty info from filename"""
        # Try to match common difficulty patterns
        difficulty_patterns = [
            r'easy|简单',
            r'medium|normal|中等|普通',
            r'hard|difficult|困难',
            r'expert|专家',
            r'easy_(\d+)',
            r'medium_(\d+)',
            r'hard_(\d+)'
        ]
        
        filename_lower = filename.lower()
        
        for pattern in difficulty_patterns:
            match = re.search(pattern, filename_lower)
            if match:
                if 'easy' in pattern or '简单' in pattern:
                    return 'easy'
                elif 'medium' in pattern or 'normal' in pattern or '中等' in pattern or '普通' in pattern:
                    return 'medium'
                elif 'hard' in pattern or 'difficult' in pattern or '困难' in pattern:
                    return 'hard'
                elif 'expert' in pattern or '专家' in pattern:
                    return 'expert'
        
        # If no specific difficulty matched, return default
        return 'unknown'
        
    def extract_difficulty_from_data_or_filename(self, data, filename):
        """Extract difficulty from data or filename, prefer the difficulty field in data"""
        # First try to get difficulty from data
        if 'difficulty' in data:
            return data['difficulty']
        
        # If not found in data, infer from filename
        return self.extract_difficulty_from_filename(filename)
        
    def analyze_episode_format(self, data, difficulty, is_stats_file=False):
        """Analyze single episode format data (bomberman_episode_*.json)"""
        player_mapping = data.get('player_mapping', {})
        episode_stats = data.get('episode_stats', {})
        
        for model_name, stats in episode_stats.items():
            player_id = str(stats.get('player_id', ''))
            
            # Update model stats (by difficulty)
            self.model_stats[model_name][difficulty]['games'] += 1
            self.model_stats[model_name][difficulty]['kills'] += stats.get('kills', 0)
            self.model_stats[model_name][difficulty]['deaths'] += stats.get('deaths', 0)
            self.model_stats[model_name][difficulty]['items'] += stats.get('items_collected', 0)
            if stats.get('won', False):
                self.model_stats[model_name][difficulty]['wins'] += 1
                
            # Only update player_id stats if not a stats file
            if not is_stats_file:
                self.player_id_stats[player_id][difficulty]['games'] += 1
                self.player_id_stats[player_id][difficulty]['kills'] += stats.get('kills', 0)
                self.player_id_stats[player_id][difficulty]['deaths'] += stats.get('deaths', 0)
                self.player_id_stats[player_id][difficulty]['items'] += stats.get('items_collected', 0)
                self.player_id_stats[player_id][difficulty]['models'][model_name] += 1
                if stats.get('won', False):
                    self.player_id_stats[player_id][difficulty]['wins'] += 1
    
    def analyze_stats_format(self, data, difficulty):
        """Analyze stats format data (bomberman_stats_*.json)"""
        models = data.get('models', {})
        wins = data.get('wins', {})
        player_stats = data.get('player_stats', {})
        
        for player_id, model_name in models.items():
            stats = player_stats.get(player_id, {})
            win_count = wins.get(player_id, 0)
            episodes = stats.get('episodes', [])
            game_count = len(episodes)
            
            # Only update model stats, not player_id stats (by difficulty)
            self.model_stats[model_name][difficulty]['games'] += game_count
            self.model_stats[model_name][difficulty]['wins'] += win_count
            self.model_stats[model_name][difficulty]['kills'] += stats.get('kills', 0)
            self.model_stats[model_name][difficulty]['deaths'] += stats.get('deaths', 0)
            self.model_stats[model_name][difficulty]['items'] += stats.get('items_collected', 0)
    
    def process_files(self, result_dir):
        """Process all JSON files in the result directory"""
        json_files = glob.glob(os.path.join(result_dir, "*.json"))
        
        for file_path in json_files:
            try:
                filename = os.path.basename(file_path)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract difficulty from data or filename
                difficulty = self.extract_difficulty_from_data_or_filename(data, filename)
                
                # Determine if it's a stats file
                is_stats_file = "stats" in filename.lower()
                
                # Determine file format type
                if 'episode_stats' in data and 'player_mapping' in data:
                    # Single episode format
                    self.analyze_episode_format(data, difficulty, is_stats_file)
                elif 'models' in data and 'player_stats' in data:
                    # Stats format
                    self.analyze_stats_format(data, difficulty)
                    
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    
    def calculate_win_rates(self, stats):
        """Calculate win rates"""
        result = {}
        for key, difficulty_data in stats.items():
            result[key] = {}
            for difficulty, data in difficulty_data.items():
                win_rate = (data['wins'] / data['games'] * 100) if data['games'] > 0 else 0
                result_data = {
                    'games': data['games'],
                    'wins': data['wins'],
                    'win_rate': round(win_rate, 2),
                    'kills': data['kills'],
                    'deaths': data['deaths'],
                    'items_collected': data['items'],
                    'kd_ratio': round(data['kills'] / max(data['deaths'], 1), 2)
                }
                
                # If there is a models field, add model occurrence stats
                if 'models' in data:
                    result_data['models'] = dict(data['models'])
                    
                result[key][difficulty] = result_data
        return result
    
    def generate_report(self, output_dir):
        """Generate statistics report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate statistics results
        model_report = self.calculate_win_rates(self.model_stats)
        player_id_report = self.calculate_win_rates(self.player_id_stats)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model statistics
        model_file = os.path.join(output_dir, f"model_stats_{timestamp}.json")
        with open(model_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model_statistics': model_report
            }, f, indent=2, ensure_ascii=False)
        
        # Save player_id statistics
        player_file = os.path.join(output_dir, f"player_id_stats_{timestamp}.json")
        with open(player_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'player_id_statistics': player_id_report
            }, f, indent=2, ensure_ascii=False)
        
        # Generate summary report
        summary_file = os.path.join(output_dir, f"summary_report_{timestamp}.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("Bomberman Game Statistics Report (By Difficulty)\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Model Statistics:\n")
            f.write("-" * 30 + "\n")
            for model, difficulty_stats in model_report.items():
                f.write(f"{model}:\n")
                for difficulty, stats in difficulty_stats.items():
                    f.write(f"  Difficulty [{difficulty}]:\n")
                    f.write(f"    Total Games: {stats['games']}\n")
                    f.write(f"    Wins: {stats['wins']}\n")
                    f.write(f"    Win Rate: {stats['win_rate']}%\n")
                    f.write(f"    Kills: {stats['kills']}\n")
                    f.write(f"    Deaths: {stats['deaths']}\n")
                    f.write(f"    K/D Ratio: {stats['kd_ratio']}\n")
                    f.write(f"    Items Collected: {stats['items_collected']}\n")
                f.write("\n")
            
            f.write("Player ID Statistics:\n")
            f.write("-" * 30 + "\n")
            for player_id, difficulty_stats in player_id_report.items():
                f.write(f"Player {player_id}:\n")
                for difficulty, stats in difficulty_stats.items():
                    f.write(f"  Difficulty [{difficulty}]:\n")
                    f.write(f"    Total Games: {stats['games']}\n")
                    f.write(f"    Wins: {stats['wins']}\n")
                    f.write(f"    Win Rate: {stats['win_rate']}%\n")
                    f.write(f"    Kills: {stats['kills']}\n")
                    f.write(f"    Deaths: {stats['deaths']}\n")
                    f.write(f"    K/D Ratio: {stats['kd_ratio']}\n")
                    f.write(f"    Items Collected: {stats['items_collected']}\n")
                    if 'models' in stats:
                        f.write(f"    Model Appearances:\n")
                        for model, count in stats['models'].items():
                            f.write(f"      {model}: {count} times\n")
                f.write("\n")
        
        print(f"Statistics report generated:")
        print(f"- Model Statistics: {model_file}")
        print(f"- Player ID Statistics: {player_file}")
        print(f"- Summary Report: {summary_file}")

def main():
    analyzer = BombermanStatsAnalyzer()
    
    # Process result files
    result_dir = ""  # Specify the directory containing result files
    output_dir = ""  # Specify the directory to save the report
    
    print("Starting analysis of Bomberman game data...")
    analyzer.process_files(result_dir)
    
    print("Generating statistics report...")
    analyzer.generate_report(output_dir)
    
    print("Analysis completed!")

if __name__ == "__main__":
    main()