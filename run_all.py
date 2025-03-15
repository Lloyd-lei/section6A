#!/usr/bin/env python3
"""
Main script to run all tasks for Physics 129AL Section Worksheet Week 6A
"""

import os
import sys
import subprocess
import time

def print_header(message):
    """Print a formatted header message"""
    print("\n" + "=" * 80)
    print(f" {message} ".center(80, "="))
    print("=" * 80 + "\n")

def run_task(task_dir, script_name):
    """Run a task script and handle errors"""
    full_path = os.path.join(task_dir, script_name)
    
    if not os.path.exists(full_path):
        print(f"Error: Script {full_path} not found!")
        return False
    
    try:
        print(f"Running {full_path}...")
        result = subprocess.run([sys.executable, full_path], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running {full_path}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error running {full_path}: {e}")
        return False

def main():
    """Main function to run all tasks"""
    print_header("Physics 129AL - Section Worksheet Week 6A")
    
    # Task 1: Poisson
    print_header("Task 1: Poisson Distribution for Random Star Distribution")
    success = run_task("task1_poisson", "poisson_stars.py")
    if not success:
        print("Task 1 failed!")
    
    # Task 2: Lorentzian
    print_header("Task 2: Lorentzian Resonance Behavior")
    success = run_task("task2_lorentzian", "lorentzian_resonance.py")
    if not success:
        print("Task 2 failed!")
    
    # Task 3: Heisenberg XXX Hamiltonian Markov Chain
    print_header("Task 3: Heisenberg XXX Hamiltonian Markov Chain Analysis")
    success = run_task("task3_heisenberg_markov", "markov_chain.py")
    if not success:
        print("Task 3 failed!")
    
    print_header("All tasks completed")
    print("Results are saved in the respective task directories.")
    print("Check the generated plots and output files for detailed results.")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds") 