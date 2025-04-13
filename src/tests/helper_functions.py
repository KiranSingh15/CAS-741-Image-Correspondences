import shutil
from datetime import datetime
from pathlib import Path


def create_timestamped_output_dir(test_id: str) -> Path:
    """
    Creates a timestamped directory under tests/Outputs and writes a summary.txt file.

    :param test_id: A string identifying the test (e.g., "STFR-IS-01").
    :return: Path to the created output directory.
    """
    base_dir = Path(__file__).parent / "Outputs"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_dir = base_dir / timestamp
    new_dir.mkdir(parents=True, exist_ok=False)

    summary_path = new_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Test ID: {test_id}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Output Directory: {new_dir.name}\n")

    return new_dir


def clear_directory(dir_path: Path):
    """
    Deletes all contents of the given directory (files and subdirectories).
    """
    if not dir_path.exists():
        print(f"Directory does not exist: {dir_path}")
        return

    for item in dir_path.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

    print(f"Cleared directory: {dir_path}")


def copy_folder_contents(src_folder: Path, targ_folder: Path):
    """
    Copies all contents from src_folder to targ_folder.
    Subdirectories are merged if they exist.

    :param src_folder: Source directory
    :param targ_folder: Target directory
    """
    src_folder = Path(src_folder)
    targ_folder = Path(targ_folder)

    for item in src_folder.iterdir():
        dest = targ_folder / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)


def summarize_image_check_results(results: list, check_type: str) -> str:
    """
    Formats results of image consistency checks.

    :param results: A list of (filename, error_type) tuples
    :param check_type: String like 'grayscale' or 'smoothed'
    :return: Summary string
    """
    if not results:
        return f"All {check_type} images passed validation.\n"

    summary = f"Issues detected in {check_type} images:\n"
    for filename, issue in results:
        summary += f" - {filename}: {issue}\n"
    return summary


def copy_selected_subfolders(
    source_dir: Path, target_dir: Path, folders_to_copy: list[str]
):
    """
    Copies only selected subfolders from source_dir to target_dir.

    :param source_dir: The base output directory (e.g., projectFiles/Outputs)
    :param target_dir: The archive directory under tests/Outputs/<timestamp>/
    :param folders_to_copy: List of folder names to selectively copy
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    for folder_name in folders_to_copy:
        src_path = source_dir / folder_name
        dest_path = target_dir / folder_name

        if src_path.exists() and src_path.is_dir():
            shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
        else:
            print(f"[Warning] Folder '{folder_name}' not found in {source_dir}")
