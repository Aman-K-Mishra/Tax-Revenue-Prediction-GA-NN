import os
import subprocess
import sys

def run(cmd):
    print("\n==================================================")
    print(f" RUNNING: {cmd}")
    print("==================================================\n")
    r = subprocess.run([sys.executable, "-m"] + cmd.split(), text=True)
    if r.returncode != 0:
        print(f"\n❌ ERROR during: {cmd}")
        sys.exit(1)

def main():
    print("\n==========================================")
    print("   TAX REVENUE PREDICTION PIPELINE START")
    print("==========================================\n")

    # 1. Run GA search
    run("scripts.run_ga_search")

    # 2. Retrain best model (GA already saved JSON)
    run("scripts.retrain_best_model")

    print("\n==========================================")
    print("   PIPELINE COMPLETED SUCCESSFULLY")
    print("==========================================\n")

    ask = input("Do you want to run prediction? (y/n): ").strip().lower()
    if ask == "y":
        run("scripts.predict")

if __name__ == "__main__":
    main()
