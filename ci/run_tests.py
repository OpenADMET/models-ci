import yaml
import subprocess
import os
import sys
from tempfile import TemporaryDirectory

OADMET_MODELS_IMAGE = "ghcr.io/openadmet/openadmet-models:main"
CONTAINER_MOUNT_POINT = "/home/mambauser/model"

def run_docker(local_path):
    cmd = [
        "docker", "run", "--rm",
        "--user", "root",
        "-v", f"{os.path.abspath(local_path)}:{CONTAINER_MOUNT_POINT}:rw",
        "--runtime", "nvidia",
        "--gpus", "all",
        OADMET_MODELS_IMAGE,
        "/bin/bash", "-c", f"cd {CONTAINER_MOUNT_POINT} && ./run_model_inference.sh"
    ]
    subprocess.run(cmd, check=True)

def run_anvil(local_path):
    # Runs anvil in the context of the cloned directory
    subprocess.run(["anvil", "run", "."], cwd=local_path, check=True)

def test_runner(models, test_type):
    """
    Generic runner to handle cloning and testing. 
    Returns True if all models pass, False otherwise.
    """
    all_passed = True
    results = {}
    
    print(f"\n{'='*20} STARTING {test_type.upper()} TESTS {'='*20}")
    
    for model in models:
        name = model["name"]
        url = model["url"]
        print(f"\n[!] Testing {name}...")
        
        try:
            with TemporaryDirectory() as tmpdir:
                # Use git clone. Note: Ensure git-lfs is installed on host for HF models
                subprocess.run(["git", "clone", "--depth", "1", url, tmpdir], 
                               check=True, capture_output=True)
                
                if test_type == "docker":
                    run_docker(tmpdir)
                else:
                    run_anvil(tmpdir)
                
                results[name] = "PASSED"
                print(f"--- {name}: OK")
        except Exception as e:
            results[name] = "FAILED"
            all_passed = False
            print(f"--- {name}: FAILED\n    Error: {e}")
            
    return all_passed, results

def main():
    if not os.path.exists("models.yaml"):
        print("Error: models.yaml not found.")
        sys.exit(1)

    with open("models.yaml", "r") as f:
        data = yaml.safe_load(f)
    
    models = data.get("models", [])
    if not models:
        print("No models found in YAML.")
        sys.exit(0)

    # LOOP 1: Docker
    docker_ok, docker_results = test_runner(models, "docker")

    # LOOP 2: Anvil
    anvil_ok, anvil_results = test_runner(models, "anvil")

    # FINAL SUMMARY TABLE
    print(f"\n\n{'='*15} FINAL TEST SUMMARY {'='*15}")
    print(f"{'Model Name':<60} | {'Docker':<10} | {'Anvil':<10}")
    print("-" * 85)
    for model in models:
        name = model["name"]
        d_res = docker_results.get(name, "N/A")
        a_res = anvil_results.get(name, "N/A")
        print(f"{name:<60} | {d_res:<10} | {a_res:<10}")

    # EXIT LOGIC
    if not docker_ok or not anvil_ok:
        print("\n[RESULT] One or more tests failed.")
        sys.exit(1)
    else:
        print("\n[RESULT] All tests passed successfully.")
        sys.exit(0)

if __name__ == "__main__":
    main()