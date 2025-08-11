import os
import re
from collections import defaultdict
from datetime import datetime
import argparse
import json

def get_model_type(filename):
    """根据文件名确定模型类型"""
    filename_lower = filename.lower()
    if 'openai' in filename_lower:
        return 'OpenAI'
    elif 'baichuan' in filename_lower:
        return 'Baichuan'
    elif 'cpm' in filename_lower:
        return 'CPM'
    elif 'qwen' in filename_lower:
        return 'Qwen'
    elif 'pro' in filename_lower:
        return 'Pro'
    else:
        return 'Unknown'

def parse_result_file(file_path):
    """解析单个结果文件，支持中英文格式"""
    data = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 提取各项数据 - 支持中英文
            # 难度级别
            difficulty_match = re.search(r'(?:难度级别|Difficulty level):\s*(\w+)', content)
            # 种子值
            seed_match = re.search(r'(?:种子值|Seed value):\s*(\d+)', content)
            # 总步数
            steps_match = re.search(r'(?:总步数|Total steps):\s*(\d+)', content)
            # 总奖励
            reward_match = re.search(r'(?:总奖励|Total reward):\s*([-+]?\d*\.?\d+)', content)
            # 无效动作次数
            invalid_match = re.search(r'(?:无效动作次数|Invalid actions):\s*(\d+)', content)
            # 成功到达目标
            success_match = re.search(r'(?:成功到达目标|Success):\s*(\S+)', content)
            # 回合被截断
            truncated_match = re.search(r'(?:回合被截断|Truncated):\s*(\S+)', content)
            # 保存时间
            time_match = re.search(r'(?:保存时间|Save time):\s*(\d+)', content)
            
            if difficulty_match:
                data['difficulty'] = difficulty_match.group(1).lower()
            if seed_match:
                data['seed'] = int(seed_match.group(1))
            if steps_match:
                data['steps'] = int(steps_match.group(1))
            if reward_match:
                data['reward'] = float(reward_match.group(1))
            if invalid_match:
                data['invalid_actions'] = int(invalid_match.group(1))
            if success_match:
                success_text = success_match.group(1).lower()
                data['success'] = success_text in ['是', 'true', 'yes']
            if truncated_match:
                truncated_text = truncated_match.group(1).lower()
                data['truncated'] = truncated_text in ['是', 'true', 'yes']
            if time_match:
                data['save_time'] = time_match.group(1)
                
            # 提取轮次信息
            round_match = re.search(r'round(\d+)', os.path.basename(file_path))
            if round_match:
                data['round'] = int(round_match.group(1))
                
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
    
    return data

def calculate_statistics(data_list):
    """计算统计数据"""
    if not data_list:
        return {}
    
    stats = {}
    stats['count'] = len(data_list)
    
    # 成功率
    success_count = sum(1 for d in data_list if d.get('success', False))
    stats['success_rate'] = success_count / len(data_list) * 100
    stats['success_count'] = success_count
    
    # 截断率
    truncated_count = sum(1 for d in data_list if d.get('truncated', False))
    stats['truncated_rate'] = truncated_count / len(data_list) * 100
    stats['truncated_count'] = truncated_count
    
    # 步数统计
    steps = [d['steps'] for d in data_list if 'steps' in d]
    if steps:
        stats['avg_steps'] = sum(steps) / len(steps)
        stats['min_steps'] = min(steps)
        stats['max_steps'] = max(steps)
        stats['median_steps'] = sorted(steps)[len(steps)//2]
        
        # 去掉最值后的平均步数
        if len(steps) > 2:
            trimmed_steps = sorted(steps)[1:-1]
            stats['trimmed_avg_steps'] = sum(trimmed_steps) / len(trimmed_steps)
    
    # 奖励统计
    rewards = [d['reward'] for d in data_list if 'reward' in d]
    if rewards:
        stats['avg_reward'] = sum(rewards) / len(rewards)
        stats['min_reward'] = min(rewards)
        stats['max_reward'] = max(rewards)
        stats['median_reward'] = sorted(rewards)[len(rewards)//2]
    
    # 无效动作统计
    invalid = [d['invalid_actions'] for d in data_list if 'invalid_actions' in d]
    if invalid:
        stats['avg_invalid'] = sum(invalid) / len(invalid)
        stats['max_invalid'] = max(invalid)
        stats['min_invalid'] = min(invalid)
    
    # 种子统计
    seeds = [d['seed'] for d in data_list if 'seed' in d]
    if seeds:
        stats['seeds'] = sorted(list(set(seeds)))
        stats['unique_seeds'] = len(set(seeds))
    
    return stats

def print_stats_for_group(stats, group_name, file_handle=None):
    """打印一组数据的统计信息"""
    if not stats:
        return
    
    output = f"\n{group_name}:\n"
    output += f"  Total games: {stats['count']}\n"
    output += f"  Success rate: {stats['success_rate']:.1f}% ({stats['success_count']}/{stats['count']})\n"
    
    if 'truncated_rate' in stats:
        output += f"  Truncated rate: {stats['truncated_rate']:.1f}% ({stats['truncated_count']}/{stats['count']})\n"
    
    if 'avg_steps' in stats:
        output += f"  Average steps: {stats['avg_steps']:.1f}\n"
        output += f"  Step range: {stats['min_steps']} - {stats['max_steps']}\n"
        output += f"  Median steps: {stats['median_steps']}\n"
        if 'trimmed_avg_steps' in stats:
            output += f"  Trimmed average steps: {stats['trimmed_avg_steps']:.1f}\n"
    
    if 'avg_reward' in stats:
        output += f"  Average reward: {stats['avg_reward']:.2f}\n"
        output += f"  Reward range: {stats['min_reward']:.2f} - {stats['max_reward']:.2f}\n"
        output += f"  Median reward: {stats['median_reward']:.2f}\n"
    
    if 'avg_invalid' in stats:
        output += f"  Average invalid actions: {stats['avg_invalid']:.1f}\n"
        output += f"  Invalid actions range: {stats['min_invalid']} - {stats['max_invalid']}\n"
    
    if 'unique_seeds' in stats:
        output += f"  Unique seeds used: {stats['unique_seeds']}\n"
        if len(stats['seeds']) <= 10:
            output += f"  Seeds: {stats['seeds']}\n"
    
    print(output, end='')
    if file_handle:
        file_handle.write(output)

def analyze_results(results_dir="results", output_format="text", export_json=False):
    """分析所有结果文件"""
    
    if not os.path.exists(results_dir):
        print(f"Directory does not exist: {results_dir}")
        return
    
    # 创建输出文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"maze_analysis_report_{timestamp}.txt"
    json_file = f"maze_analysis_data_{timestamp}.json" if export_json else None
    
    all_data = []
    model_stats = defaultdict(list)
    difficulty_stats = defaultdict(list)
    model_difficulty_stats = defaultdict(lambda: defaultdict(list))
    
    # 读取所有文件
    for filename in os.listdir(results_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(results_dir, filename)
            if os.path.isfile(file_path):
                data = parse_result_file(file_path)
                if data:
                    model_type = get_model_type(filename)
                    data['model'] = model_type
                    data['filename'] = filename
                    
                    all_data.append(data)
                    model_stats[model_type].append(data)
                    
                    if 'difficulty' in data:
                        difficulty_stats[data['difficulty']].append(data)
                        model_difficulty_stats[model_type][data['difficulty']].append(data)
    
    if not all_data:
        print("No valid result files found")
        return
    
    # 计算总体统计
    overall_stats = calculate_statistics(all_data)
    
    # 导出JSON数据
    if export_json:
        json_data = {
            'overall': overall_stats,
            'by_model': {model: calculate_statistics(data_list) for model, data_list in model_stats.items()},
            'by_difficulty': {diff: calculate_statistics(data_list) for diff, data_list in difficulty_stats.items()},
            'by_model_difficulty': {
                model: {diff: calculate_statistics(data_list) for diff, data_list in diff_stats.items()}
                for model, diff_stats in model_difficulty_stats.items()
            },
            'raw_data': all_data
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"JSON data exported to: {json_file}")
    
    # 同时输出到控制台和文件
    with open(output_file, 'w', encoding='utf-8') as f:
        # 输出统计结果
        header = "=" * 60 + "\nMaze Navigation Analysis Report\n" + "=" * 60 + "\n"
        summary = f"Total files analyzed: {len(all_data)}\n"
        summary += f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        print(header + summary)
        f.write(header + summary)
        
        # 总体统计
        print_stats_for_group(overall_stats, "OVERALL STATISTICS", f)
        
        # 按模型统计
        model_header = "\n" + "=" * 40 + "\nStatistics by Model\n" + "=" * 40
        print(model_header)
        f.write(model_header)
        
        for model_type in sorted(model_stats.keys()):
            model_data = model_stats[model_type]
            stats = calculate_statistics(model_data)
            print_stats_for_group(stats, f"{model_type} Model", f)
        
        # 按难度级别统计
        difficulty_header = "\n" + "=" * 40 + "\nStatistics by Difficulty\n" + "=" * 40
        print(difficulty_header)
        f.write(difficulty_header)
        
        for difficulty in ['easy', 'medium', 'hard']:
            if difficulty in difficulty_stats:
                data_list = difficulty_stats[difficulty]
                stats = calculate_statistics(data_list)
                print_stats_for_group(stats, f"{difficulty.upper()} Difficulty", f)
        
        # 按模型和难度交叉统计
        cross_header = "\n" + "=" * 40 + "\nCross Statistics (Model × Difficulty)\n" + "=" * 40
        print(cross_header)
        f.write(cross_header)
        
        for model_type in sorted(model_difficulty_stats.keys()):
            model_detail_header = f"\n{model_type} Model Detailed Statistics:"
            print(model_detail_header)
            f.write(model_detail_header)
            
            for difficulty in ['easy', 'medium', 'hard']:
                if difficulty in model_difficulty_stats[model_type]:
                    data_list = model_difficulty_stats[model_type][difficulty]
                    stats = calculate_statistics(data_list)
                    print_stats_for_group(stats, f"  {difficulty.upper()} Difficulty", f)
    
    print(f"\nAnalysis report saved to: {output_file}")

def main():
    """主函数，支持命令行参数"""
    parser = argparse.ArgumentParser(description="Maze Navigation Results Analyzer")
    parser.add_argument("--results-dir", type=str, default="results", help="Results directory path")
    parser.add_argument("--export-json", action="store_true", help="Export data to JSON file")
    parser.add_argument("--format", type=str, default="text", choices=["text", "json"], help="Output format")
    
    args = parser.parse_args()
    
    analyze_results(
        results_dir=args.results_dir,
        output_format=args.format,
        export_json=args.export_json
    )

if __name__ == "__main__":
    main()
