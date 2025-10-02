# main.py - Main entry point for the Streamlit app
import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'scikit-learn',
        'joblib',
        'plotly',
        'Pillow'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_missing_packages(missing_packages):
    """Install missing packages"""
    if missing_packages:
        print(f"Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Successfully installed {package}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
                return False
    return True

def check_model_files():
    """Check if required model files exist"""
    required_files = [
        'best_energy_model.pkl',
        'preprocessing_artifacts.pkl',
        'model_metadata.json'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    return missing_files

def main():
    """Main function to run the Streamlit app"""
    print("🚀 Energy Consumption Predictor - Starting Application...")
    
    # Check dependencies
    print("📦 Checking dependencies...")
    missing_packages = check_dependencies()
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        if install_missing_packages(missing_packages):
            print("✅ All dependencies installed successfully!")
        else:
            print("❌ Failed to install some dependencies. Please install manually.")
            sys.exit(1)
    else:
        print("✅ All dependencies are available!")
    
    # Check model files
    print("🔍 Checking model files...")
    missing_files = check_model_files()
    
    if missing_files:
        print(f"❌ Missing model files: {', '.join(missing_files)}")
        print("Please ensure you have trained the model and have these files in the current directory:")
        print("  - best_energy_model.pkl")
        print("  - preprocessing_artifacts.pkl")
        print("  - model_metadata.json")
        print("\nRun the training script first to generate these files.")
        sys.exit(1)
    else:
        print("✅ All model files are available!")
    
    # Launch Streamlit app
    print("🌐 Launching Streamlit application...")
    print("📍 The app will open in your default browser shortly...")
    print("📱 You can also access it at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    print("\n" + "="*50)
    
    try:
        # Run Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--theme.primaryColor", "#1f77b4",
            "--theme.backgroundColor", "#ffffff",
            "--theme.secondaryBackgroundColor", "#f0f2f6",
            "--theme.textColor", "#262730"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()