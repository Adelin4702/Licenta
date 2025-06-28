import subprocess
import sys

def install_packages(packages):
    """Function to install a list of packages using pip"""
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"? Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"? Failed to install {package}")

if __name__ == "__main__":
    packages = [
        "ujson",
        "utils",
        "albumentations",  
        "Pillow", 
        "scikit-image",
        "filterpy",
        "opencv-fixer==0.2.5"
    ]

    print(" Installing required packages...")
    install_packages(packages)
    print(" All packages installed successfully!")
    
    print("\n\nRunning opencv_fixer.AutoFix()...")
    try:
        subprocess.check_call([
            sys.executable,
            "-c",
            "from opencv_fixer import AutoFix; AutoFix()"
        ])
        print("✅ AutoFix ran successfully!")
    except subprocess.CalledProcessError:
        print("❌ Failed to run AutoFix.")

