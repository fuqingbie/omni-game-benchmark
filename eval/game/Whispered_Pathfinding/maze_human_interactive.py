import gym
import numpy as np
import pygame
import base64
import io
import os
import tempfile
import time
import datetime
import pathlib
from maze_gym_env import MazeGymEnv

class HumanMazeRunner:
    """人类操作的迷宫运行器，允许用户通过终端输入动作来控制角色"""
    
    def __init__(self, difficulty='easy', max_steps=500):
        # 初始化Pygame（用于音频播放）
        pygame.init()
        pygame.mixer.init()
        
        # 存储游戏设置
        self.difficulty = difficulty
        self.max_steps = max_steps
        
        # 创建环境，启用语音导航，设置难度
        self.env = MazeGymEnv(render_mode="human", voice_guidance=True, difficulty=self.difficulty)
        
        # 创建临时目录存储音频文件
        self.temp_dir = tempfile.mkdtemp(prefix="human_maze_")
        
        # 跟踪总奖励和当前步骤
        self.total_reward = 0
        self.current_step = 0
        
        # 游戏统计
        self.stats = {
            "steps": 0,
            "total_reward": 0,
            "invalid_actions": 0
        }
    
    def play_audio(self, audio_base64):
        """播放音频提示"""
        if not audio_base64:
            return
            
        try:
            # 将base64解码为音频文件
            audio_data = base64.b64decode(audio_base64)
            audio_file = os.path.join(self.temp_dir, "current_guidance.wav")
            
            with open(audio_file, "wb") as f:
                f.write(audio_data)
            
            # 播放音频
            sound = pygame.mixer.Sound(audio_file)
            sound.play()
            
            # 等待音频播放完成（可选）
            pygame.time.wait(int(sound.get_length() * 1000))
        except Exception as e:
            print(f"音频播放错误: {e}")
    
    def get_user_input(self):
        """获取用户输入的动作"""
        print("\n===== 请输入动作 =====")
        print("动作格式: [前进距离] [旋转角度]")
        print("前进距离范围: [-1.0, 3.0]，负值表示后退")
        print("旋转角度范围: [-180.0, 180.0]，负值表示向左转")
        print("示例: '1.0 45' 表示前进1.0个单位并向右旋转45度")
        print("输入'q'退出游戏")
        print("输入'r'重置游戏")
        print("输入'd'更改难度")
        
        while True:
            try:
                user_input = input("> ")
                
                # 检查特殊命令
                if user_input.lower() == 'q':
                    return None
                elif user_input.lower() == 'r':
                    return 'reset'
                elif user_input.lower() == 'd':
                    return 'change_difficulty'
                    
                # 解析输入
                values = user_input.split()
                if len(values) != 2:
                    print("错误: 请输入两个值，用空格分隔")
                    continue
                    
                forward = float(values[0])
                rotation = float(values[1])
                
                # 验证范围
                if not (-1.0 <= forward <= 3.0):
                    print("错误: 前进距离必须在[-1.0, 3.0]范围内")
                    self.stats["invalid_actions"] += 1
                    continue
                    
                if not (-180.0 <= rotation <= 180.0):
                    print("错误: 旋转角度必须在[-180.0, 180.0]范围内")
                    self.stats["invalid_actions"] += 1
                    continue
                    
                # 返回有效的动作
                return np.array([forward, rotation], dtype=np.float32)
                
            except ValueError:
                print("错误: 请输入有效的数值")
                self.stats["invalid_actions"] += 1
                continue
    
    def print_observation_info(self, observation):
        """打印观察信息"""
        vector_obs = observation['vector']
        
        print("\n===== 当前状态 =====")
        print(f"难度: {self.difficulty}")
        print(f"位置: ({vector_obs[0]:.2f}, {vector_obs[1]:.2f})")
        print(f"朝向: {np.degrees(vector_obs[2]):.1f}°")
        print(f"到目标距离: {vector_obs[3]:.2f}米")
        print(f"到目标角度: {np.degrees(vector_obs[4]):.1f}°")
        
        # 输出主要的墙壁距离信息
        directions = ["前方", "右前方", "右侧", "右后方", "后方", "左后方", "左侧", "左前方"]
        for i, direction in enumerate(directions):
            print(f"{direction}墙壁距离: {vector_obs[5+i]:.2f}米")
    
    def change_difficulty(self):
        """更改游戏难度"""
        print("\n选择新难度:")
        print("1. 简单 (Easy)")
        print("2. 中等 (Medium)")
        print("3. 困难 (Hard)")
        
        while True:
            choice = input("请选择 (1-3): ")
            if choice == "1":
                new_difficulty = "easy"
                break
            elif choice == "2":
                new_difficulty = "medium"
                break
            elif choice == "3":
                new_difficulty = "hard"
                break
            else:
                print("无效选择，请重新输入")
        
        # 更新难度并重置环境
        self.difficulty = new_difficulty
        observation, info = self.env.reset()
        if observation:
            print(f"\n难度已更改为: {new_difficulty}")
            self.total_reward = 0
            self.current_step = 0
            return observation, info
        return None, None
    
    def save_stats(self, round_num, terminated, truncated):
        """保存统计信息到文件"""
        # 创建保存目录（如果不存在）
        save_dir = pathlib.Path("")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取当前时间作为文件名一部分
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
        # 确定当前的代码文件名（不包含.py扩展名）
        script_name = os.path.basename(__file__).replace('.py', '')
        
        # 创建文件名
        filename = f"{script_name}_round{round_num}_{timestamp}.txt"
        filepath = save_dir / filename
        
        # 写入统计信息
        with open(filepath, 'w') as f:
            f.write("===== 导航统计 =====\n")
            f.write(f"难度级别: {self.difficulty}\n")
            f.write(f"总步数: {self.stats['steps']}\n")
            f.write(f"总奖励: {self.stats['total_reward']:.2f}\n")
            f.write(f"无效动作次数: {self.stats['invalid_actions']}\n")
            f.write(f"成功到达目标: {'是' if terminated else '否'}\n")
            if truncated:
                f.write("回合被截断: 是\n")
            f.write(f"保存时间: {timestamp}\n")
        
        print(f"统计信息已保存到: {filepath}")

    def run_single_round(self, round_num):
        """运行单回合迷宫导航"""
        print(f"\n===== 开始第 {round_num} 回合 =====")
        print(f"难度级别: {self.difficulty}")
        
        # 重置环境
        observation, info = self.env.reset()
        
        # 重置统计信息
        self.total_reward = 0
        self.current_step = 0
        self.stats = {
            "steps": 0,
            "total_reward": 0,
            "invalid_actions": 0
        }
        
        # 显示初始状态信息
        self.print_observation_info(observation)
        
        # 播放初始音频提示
        if 'audio' in info:
            print("播放语音导航...")
            self.play_audio(info['audio'])
        
        # 交互式循环
        done = False
        while not done and self.current_step < self.max_steps:
            # 获取用户输入的动作
            action = self.get_user_input()
            
            # 检查特殊命令 - 使用isinstance进行类型检查
            if action is None:  # 用户选择退出
                return False, False
            elif isinstance(action, str) and action == 'reset':  # 用户选择重置
                observation, info = self.env.reset()
                self.total_reward = 0
                self.current_step = 0
                self.print_observation_info(observation)
                if 'audio' in info:
                    self.play_audio(info['audio'])
                continue
            elif isinstance(action, str) and action == 'change_difficulty':  # 用户选择更改难度
                observation, info = self.change_difficulty()
                if observation:
                    self.print_observation_info(observation)
                    if 'audio' in info:
                        self.play_audio(info['audio'])
                continue
            
            # 执行动作 (此时action是numpy数组)
            observation, reward, terminated, truncated, info = self.env.step(action)
            self.total_reward += reward
            self.current_step += 1
            
            # 更新环境显示
            self.env.render()
            
            # 显示状态信息
            print(f"\n步骤 {self.current_step}/{self.max_steps}, 总奖励: {self.total_reward:.2f}")
            print(f"执行动作: 前进={action[0]:.2f}, 旋转={action[1]:.2f}°")
            print(f"到目标距离: {observation['vector'][3]:.2f}米")
            
            # 播放音频提示(如果有)
            if 'audio' in info:
                print("正在播放语音导航...")
                self.play_audio(info['audio'])
            
            # 检查是否结束
            done = terminated or truncated
            if terminated:
                print("\n恭喜! 成功到达目标!")
            elif truncated:
                print("\n回合结束!")
        
        # 更新并显示统计信息
        self.stats["steps"] = self.current_step
        self.stats["total_reward"] = self.total_reward
        
        print("\n===== 导航统计 =====")
        print(f"难度级别: {self.difficulty}")
        print(f"总步数: {self.stats['steps']}")
        print(f"总奖励: {self.stats['total_reward']:.2f}")
        print(f"无效动作次数: {self.stats['invalid_actions']}")
        print(f"成功到达目标: {'是' if terminated else '否'}")
        
        # 保存统计信息
        self.save_stats(round_num, terminated, truncated)
        
        return terminated, truncated
    
    def run(self):
        """运行多回合迷宫导航"""
        print(f"启动人类控制迷宫导航 - 难度: {self.difficulty}")
        print(f"最大步数: {self.max_steps}")
        print(f"将运行5个回合，每回合结束后会保存统计数据")
        
        total_rounds = 5
        completed_rounds = 0
        success_rounds = 0
        
        try:
            for round_num in range(1, total_rounds + 1):
                terminated, truncated = self.run_single_round(round_num)
                completed_rounds += 1
                if terminated:
                    success_rounds += 1
                
                # 询问是否继续
                if round_num < total_rounds:
                    print(f"\n是否继续下一回合? (y/n)")
                    choice = input("> ")
                    if choice.lower() != 'y':
                        break
            
            # 打印总体统计
            print("\n===== 全部回合统计 =====")
            print(f"完成回合数: {completed_rounds}/{total_rounds}")
            print(f"成功到达目标回合数: {success_rounds}/{total_rounds}")
            print(f"成功率: {(success_rounds/total_rounds)*100:.1f}%")
            
        except KeyboardInterrupt:
            print("\n用户中断测试")
            print(f"已完成回合数: {completed_rounds}/{total_rounds}")
        finally:
            # 清理资源
            self.env.close()
            pygame.quit()
            
            # 清理临时文件
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"清理临时文件时出错: {e}")
            
            print("\n人类导航测试结束!")

def select_difficulty():
    """让用户选择迷宫难度"""
    print("请选择迷宫难度:")
    print("1. 简单 (Easy) - 墙壁较少，广阔路径")
    print("2. 中等 (Medium) - 适当障碍物，中等难度路径")
    print("3. 困难 (Hard) - 障碍物密集，更窄通道，更大迷宫")
    
    while True:
        choice = input("请输入选项 (1-3): ")
        if choice == "1":
            return "easy"
        elif choice == "2":
            return "medium"
        elif choice == "3":
            return "hard"
        else:
            print("无效的选择，请重新输入。")

if __name__ == "__main__":
    # 选择难度级别
    difficulty = select_difficulty()
    
    # 创建并运行人类迷宫
    runner = HumanMazeRunner(difficulty=difficulty, max_steps=500)
    runner.run()
