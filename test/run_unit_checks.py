import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Set up paths
this_dir = Path(__file__).resolve().parent

# Create a timestamped folder inside /results
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_dir = this_dir / "results" / timestamp
results_dir.mkdir(parents=True, exist_ok=True)

# List of test files to run
test_files = [
    "test_specParams.py",
    "test_config.py",
    "test_imagesmooth.py",
    "test_kpdetect.py",
    "test_featdesc.py",
    "test_featmatches.py",
    "test_imagePlot.py",
    "test_outputFormat.py",
    "test_verifyOutput.py",
    "test_main.py",
]

def run_test_and_save_output(test_file):
    test_path = this_dir / test_file
    result_path = results_dir / f"{test_file.replace('.py', '')}_results.txt"

    with open(result_path, "w") as outfile:
        print(f"Running {test_file} ...")
        return_code = subprocess.call(
            [sys.executable, "-m", "pytest", "-v", str(test_path)],
            stdout=outfile,
            stderr=subprocess.STDOUT,
        )

    return test_file, return_code

def summarize_results(results_dir, timestamp, test_results):
    summary_path = results_dir / "summary.txt"
    with open(summary_path, "w") as summary_file:
        summary_file.write(f"Unit Test Summary ({timestamp})\n")
        summary_file.write("=" * 60 + "\n\n")

        for test_file, result_code in test_results:
            result_txt_path = results_dir / f"{test_file.replace('.py', '')}_results.txt"

            summary_file.write(f"Program: {test_file}\n")

            # Read raw pytest output
            with open(result_txt_path, "r") as f:
                lines = f.readlines()

            # Count total, passed, failed
            test_lines = [l for l in lines if "::" in l and ("PASSED" in l or "FAILED" in l)]
            passed = sum("PASSED" in l for l in test_lines)
            failed = sum("FAILED" in l for l in test_lines)
            total = passed + failed

            summary_file.write(f"  Total tests: {total}\n")
            summary_file.write(f"  Passed: {passed}\n")
            summary_file.write(f"  Failed: {failed}\n")

            if failed > 0:
                summary_file.write("  Failed Tests:\n")
                for line in test_lines:
                    if "FAILED" in line:
                        summary_file.write(f"    - {line.strip()}\n")

            summary_file.write("\n")

def main():
    results = []
    for test_file in test_files:
        name, code = run_test_and_save_output(test_file)
        results.append((name, code))

    print("\n--- Test Summary ---")
    for name, code in results:
        status = "PASSED" if code == 0 else "FAILED"
        print(f"{name}: {status} (results saved to: results/{timestamp}/{name.replace('.py', '')}_results.txt)")

    # Write the summary file inside main(), while `results` is still in scope
    summarize_results(results_dir, timestamp, results)




if __name__ == "__main__":
    main()
