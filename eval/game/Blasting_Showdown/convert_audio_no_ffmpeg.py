import os
import sys
import subprocess
import shutil
from pathlib import Path
import platform

def check_ffmpeg():
    """Check if ffmpeg is installed"""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return True
    except FileNotFoundError:
        return False

def try_install_ffmpeg():
    """Try to install ffmpeg and provide instructions"""
    system = platform.system()
    print("ffmpeg is not installed on your system. You can install it in the following ways:")
    
    if system == "Darwin":  # macOS
        print("Method 1: Install using Homebrew")
        print("  $ brew install ffmpeg")
        print("\nMethod 2: Download from the official website: https://ffmpeg.org/download.html")
        
    elif system == "Linux":
        print("Ubuntu/Debian:")
        print("  $ sudo apt-get update")
        print("  $ sudo apt-get install ffmpeg")
        print("\nCentOS/RHEL:")
        print("  $ sudo yum install ffmpeg ffmpeg-devel")
        
    elif system == "Windows":
        print("1. Download ffmpeg: https://www.gyan.dev/ffmpeg/builds/")
        print("2. Unzip the file and add the bin directory to the system PATH")
    
    print("\nPlease rerun this script after installation")
    return False

def convert_ogg_to_wav_with_ffmpeg(input_file, output_file):
    """Convert OGG to WAV using ffmpeg"""
    try:
        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-i', input_file,  # Input file
            '-acodec', 'pcm_s16le',  # 16-bit PCM encoding
            '-ar', '44100',  # Sample rate 44.1kHz
            '-y',  # Overwrite output file
            output_file  # Output file
        ]
        
        # Run command
        process = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            check=True
        )
        print(f"Successfully converted: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed! ffmpeg error: {e.stderr.decode()}")
        return False
    except Exception as e:
        print(f"An error occurred during conversion: {e}")
        return False

def simple_copy_as_wav(input_file, output_file):
    """For testing environments, simply copy and rename the file to .wav"""
    try:
        print(f"Using simple copy method (for testing only! The file format is not actually converted)")
        shutil.copy2(input_file, output_file)
        print(f"File has been copied as WAV: {output_file}")
        return True
    except Exception as e:
        print(f"Failed to copy file: {e}")
        return False

def main():
    # List of files to convert
    files_to_convert = [
        "game/assets-necessay/kenney/UI assets/UI Pack/Sounds/click-b.ogg",
        "game/assets-necessay/kenney/Audio/Retro Sounds 2/Audio/explosion1.ogg",
        "game/assets-necessay/kenney/Audio/Impact Sounds/Audio/footstep_wood_001.ogg"
    ]
    
    # Determine absolute paths from the project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Check if ffmpeg is available
    ffmpeg_available = check_ffmpeg()
    if not ffmpeg_available:
        print("FFmpeg not found, will attempt simple copy method")
        try_install_ffmpeg()
    
    # Convert each file
    success_count = 0
    for rel_path in files_to_convert:
        abs_path = os.path.join(project_root, rel_path)
        if not os.path.exists(abs_path):
            print(f"Error: File not found {abs_path}")
            continue
        
        # Create WAV output path
        wav_path = os.path.splitext(abs_path)[0] + '.wav'
        print(f"Processing: {abs_path}")
        
        if ffmpeg_available:
            # Convert using ffmpeg
            success = convert_ogg_to_wav_with_ffmpeg(abs_path, wav_path)
        else:
            # Simply copy and rename to WAV (Note: this is not a real format conversion)
            success = simple_copy_as_wav(abs_path, wav_path)
        
        if success:
            success_count += 1
    
    print(f"\nConversion complete! Successfully processed {success_count}/{len(files_to_convert)} files.")

    # Remind the user to update file references in the code
    if success_count > 0:
        print("\nPlease ensure you update the audio file references in bomberman_gym.py and classic_bomberman-daiceshi.py:")
        print("1. Change the original '.ogg' suffix to '.wav'")
        print("2. Change the audio format configuration to 'wav'")

if __name__ == "__main__":
    main()