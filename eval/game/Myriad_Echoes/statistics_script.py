import os
import re
from collections import defaultdict
import pandas as pd
from datetime import datetime

def identify_model(filename):
    """Identify model type based on filename"""
    filename_lower = filename.lower()
    if 'baichuan' in filename_lower:
        return 'baichuan-omni'
    elif 'cpm' in filename_lower:
        return 'cpm'
    elif 'flash' in filename_lower:
        return 'gemini-2.5-flash'
    elif 'pro' in filename_lower:
        return 'gemini-2.5-pro'
    elif 'qwen' in filename_lower:
        return 'qwen'
    else:
        return 'unknown'

def parse_result_file(file_path):
    """Parse result file and extract statistics"""
    data = {
        'difficulty': None,
        'total_episodes': 0,
        'success_episodes': 0,
        'success_rate': 0.0,
        'total_score': 0,
        'avg_score': 0.0,
        'unparseable_sequences': 0,
        'total_coordinate_correct': 0,
        'total_icon_correct': 0,
        'total_coordinate_attempts': 0,
        'total_icon_attempts': 0
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract basic statistics
        difficulty_match = re.search(r'Difficulty Level:\s*(\d+)', content)
        if difficulty_match:
            data['difficulty'] = int(difficulty_match.group(1))
        
        total_episodes_match = re.search(r'Total Episodes:\s*(\d+)', content)
        if total_episodes_match:
            data['total_episodes'] = int(total_episodes_match.group(1))
        
        success_episodes_match = re.search(r'Successful Episodes:\s*(\d+)', content)
        if success_episodes_match:
            data['success_episodes'] = int(success_episodes_match.group(1))
        
        success_rate_match = re.search(r'Success Rate:\s*([\d.]+)%', content)
        if success_rate_match:
            data['success_rate'] = float(success_rate_match.group(1))
        
        total_score_match = re.search(r'Total Score:\s*(\d+)', content)
        if total_score_match:
            data['total_score'] = int(total_score_match.group(1))
        
        avg_score_match = re.search(r'Average Score:\s*([\d.]+)', content)
        if avg_score_match:
            data['avg_score'] = float(avg_score_match.group(1))
        
        unparseable_match = re.search(r'Unparseable Sequences:\s*(\d+)', content)
        if unparseable_match:
            data['unparseable_sequences'] = int(unparseable_match.group(1))
        
        # Parse detailed results to count correct coordinates and icons
        episode_pattern = r'Episode \d+:.*?Correct Coordinates=(\d+)/(\d+).*?Correct Icons=(\d+)/(\d+)'
        episodes = re.findall(episode_pattern, content)
        
        for coord_correct, coord_total, icon_correct, icon_total in episodes:
            data['total_coordinate_correct'] += int(coord_correct)
            data['total_coordinate_attempts'] += int(coord_total)
            data['total_icon_correct'] += int(icon_correct)
            data['total_icon_attempts'] += int(icon_total)
        
        return data
    
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return None

def main():
    results_dir = "results"
    
    # Store all data, grouped by model and difficulty
    model_data = defaultdict(lambda: defaultdict(list))
    
    # Scan directory
    for filename in os.listdir(results_dir):
        if filename.startswith('rhythm_memory') or not filename.endswith('.txt'):
            continue
        
        file_path = os.path.join(results_dir, filename)
        model_type = identify_model(filename)
        
        if model_type == 'unknown':
            print(f"Skipping unknown model file: {filename}")
            continue
        
        data = parse_result_file(file_path)
        if data and data['difficulty'] is not None:
            model_data[model_type][data['difficulty']].append(data)
            print(f"Processing file: {filename} -> {model_type}, Difficulty: {data['difficulty']}")
    
    # Create output directory
    output_dir = "Desktop/0611-test/statistics_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save overall statistics report
    report_file = os.path.join(output_dir, f"rhythm_memory_statistics_{timestamp}.txt")
    excel_file = os.path.join(output_dir, f"rhythm_memory_statistics_{timestamp}.xlsx")
    
    all_tables = {}
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("Rhythm Memory Game Model Performance Statistics Report\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Generate statistics table for each difficulty level
        for difficulty in [1, 2, 3]:
            print(f"\n{'='*50}")
            print(f"Difficulty Level {difficulty} Statistics")
            print(f"{'='*50}")
            
            f.write(f"Difficulty Level {difficulty} Statistics\n")
            f.write("-"*50 + "\n")
            
            if not any(difficulty in difficulties for difficulties in model_data.values()):
                print(f"No data found for difficulty level {difficulty}")
                f.write(f"No data found for difficulty level {difficulty}\n\n")
                continue
            
            # Prepare table data
            table_data = []
            
            for model_type in ['baichuan-omni', 'cpm', 'gemini-2.5-flash', 'gemini-2.5-pro', 'qwen']:
                if difficulty not in model_data[model_type]:
                    continue
                
                files_data = model_data[model_type][difficulty]
                
                # Calculate averages
                total_episodes = sum(d['total_episodes'] for d in files_data)
                total_success_episodes = sum(d['success_episodes'] for d in files_data)
                total_score = sum(d['total_score'] for d in files_data)
                total_unparseable = sum(d['unparseable_sequences'] for d in files_data)
                total_coord_correct = sum(d['total_coordinate_correct'] for d in files_data)
                total_coord_attempts = sum(d['total_coordinate_attempts'] for d in files_data)
                total_icon_correct = sum(d['total_icon_correct'] for d in files_data)
                total_icon_attempts = sum(d['total_icon_attempts'] for d in files_data)
                
                avg_success_rate = (total_success_episodes / total_episodes * 100) if total_episodes > 0 else 0
                avg_score = total_score / total_episodes if total_episodes > 0 else 0
                avg_coord_correct = total_coord_correct / total_episodes if total_episodes > 0 else 0
                avg_icon_correct = total_icon_correct / total_episodes if total_episodes > 0 else 0
                unparseable_ratio = (total_unparseable / total_episodes * 100) if total_episodes > 0 else 0
                
                table_data.append({
                    'Model': model_type,
                    'File Count': len(files_data),
                    'Total Episodes': total_episodes,
                    'Avg Success Rate (%)': f"{avg_success_rate:.2f}",
                    'Avg Score': f"{avg_score:.2f}",
                    'Avg Correct Coordinates': f"{avg_coord_correct:.2f}",
                    'Avg Correct Icons': f"{avg_icon_correct:.2f}",
                    'Unparseable Sequence Ratio (%)': f"{unparseable_ratio:.2f}"
                })
            
            # Create and display table
            if table_data:
                df = pd.DataFrame(table_data)
                print(df.to_string(index=False))
                f.write(df.to_string(index=False))
                f.write("\n\n")
                
                # Save to different sheets in Excel
                all_tables[f'Difficulty_{difficulty}'] = df
            else:
                print("No data to display")
                f.write("No data to display\n\n")
    
    # Save Excel file
    if all_tables:
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            for sheet_name, df in all_tables.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"\nStatistics saved to:")
    print(f"Text Report: {report_file}")
    print(f"Excel File: {excel_file}")

if __name__ == "__main__":
    main()