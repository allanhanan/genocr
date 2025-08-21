import requests
from pathlib import Path

def get_repo_file_list(repo_id: str, subfolder: str = "") -> list:
    """Fetches a list of file paths from a Hugging Face repository."""
    api_url = f"https://huggingface.co/api/models/{repo_id}/tree/main/{subfolder}"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        return [item['path'] for item in response.json() if item['type'] == 'file']
    except requests.exceptions.RequestException as e:
        print(f"Error fetching file list for {repo_id}: {e}")
        return []

def download_file(url: str, local_path: Path):
    """Downloads a single file with streaming."""
    print(f"Downloading {local_path.name}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"  -> Could not download {local_path.name}. Error: {e}")

def main():
    """
    Downloads all necessary files for the Phi-3 Vision corrector model:
    1. The pre-converted ONNX model files.
    2. The required support files (.py, .json) for the 'optimum' library.
    """
    # --- Path Setup ---
    project_root = Path(__file__).resolve().parent.parent.parent
    corrector_dir = project_root / "models" / "corrector"
    corrector_dir.mkdir(exist_ok=True, parents=True)

    # --- Part 1: Download ONNX Model Files ---
    print("--- Part 1: Downloading ONNX model files ---")
    onnx_repo_id = "microsoft/Phi-3-vision-128k-instruct-onnx"
    onnx_model_folder = "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
    
    onnx_files = get_repo_file_list(onnx_repo_id, onnx_model_folder)
    for file_path in onnx_files:
        url = f"https://huggingface.co/{onnx_repo_id}/resolve/main/{file_path}"
        local_path = corrector_dir / Path(file_path).name
        download_file(url, local_path)

    # --- Part 2: Download Support Files (Configs & Code) ---
    print("\n--- Part 2: Downloading required support files ---")
    pytorch_repo_id = "microsoft/Phi-3-vision-128k-instruct"
    
    support_files = get_repo_file_list(pytorch_repo_id)
    files_to_download = [f for f in support_files if f.endswith(('.py', '.json'))]

    for file_path in files_to_download:
        url = f"https://huggingface.co/{pytorch_repo_id}/resolve/main/{file_path}"
        local_path = corrector_dir / Path(file_path).name
        download_file(url, local_path)

    print("\nCorrector model setup complete.")
    print(f"All files saved to: {corrector_dir}")

if __name__ == "__main__":
    main()