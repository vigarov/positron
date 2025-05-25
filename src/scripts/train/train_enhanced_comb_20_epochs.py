import subprocess
import sys
import os
from pathlib import Path
import argparse

def get_python_executable(venv_path_str=None):
    if venv_path_str:
        venv_path = Path(venv_path_str)
        if os.name == 'nt': # Windows
            python_exec = venv_path / "Scripts" / "python.exe"
        else: # Unix-like (Linux, macOS)
            python_exec = venv_path / "bin" / "python"
        
        if not python_exec.is_file():
            print(f"Error: Python executable not found at {python_exec}. Please check the venv path.")
            sys.exit(1)
        return str(python_exec)
    return sys.executable

def main():
    parser = argparse.ArgumentParser(description="Train a model using main_trainer.py, optionally with a specific venv.")
    parser.add_argument("--venv", type=str, help="Path to the virtual environment directory to use.")
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parent.parent.parent.parent
    trainer_script_relative = Path("src") / "nn" / "main_trainer.py"
    trainer_script_absolute = workspace_root / trainer_script_relative

    python_executable = get_python_executable(args.venv)

    # Train parameters
    model_type = "enhanced_comb"
    epochs = 20
    model_name = "wb_enhcomb"

    print(f"Starting training for {model_type} model with {epochs} epochs...")
    print(f"Workspace root resolved to: {workspace_root}")
    print(f"Trainer script path: {trainer_script_absolute}")
    print(f"Python executable: {python_executable}")

    cmd_list = [
        python_executable,
        str(trainer_script_absolute),
        "--model_type", model_type,
        "--train",
        "--epochs", str(epochs),
    ]

    if 'model_name' in locals() and model_name:
        cmd_list.extend(["--model_name", model_name])

    print(f"Executing command: {' '.join(cmd_list)}")

    try:
        process = subprocess.run(
            cmd_list,
            check=True,
            text=True,
            cwd=workspace_root
        )
        print("Training script finished successfully.")
        if process.stdout:
            print("Output:\n", process.stdout)
        if process.stderr: 
            print("Error output (if any):\n", process.stderr)

    except subprocess.CalledProcessError as e:
        print(f"Training script failed with error code {e.returncode}")
        if e.stdout:
            print(f"Stdout:\n{e.stdout}")
        if e.stderr:
            print(f"Stderr:\n{e.stderr}")
    except FileNotFoundError:
        print(f"Error: The Python interpreter '{python_executable}' or trainer script '{trainer_script_absolute}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main() 