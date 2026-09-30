import io
import json
import os
import pathlib
import re
import tarfile
import uuid

import docker
from docker.errors import DockerException, ImageNotFound

SQL_RESULTS_DIR = os.getenv(
    "DATASTORM_SQL_RESULTS_DIR",
    str(pathlib.Path(__file__).resolve().parents[3] / "sql_results"),
)

_DOCKER_IMAGE = "python:3.10"

_INSTALL_CMD = "pip install -q pandas==2.2.3 plotly"

_PIP_WARNING_PATTERNS = [
    re.compile(r".*WARNING: Running pip as the 'root' user.*"),
    re.compile(r".*\[notice\] A new release of pip is available.*"),
    re.compile(r".*\[notice\] To update, run: pip install --upgrade pip"),
]

def _clean_stdout(stdout: str) -> str:
    """Remove initial pip installation output"""
    marker = "threadpoolctl-3.6.0\n"
    if marker in stdout:
        return stdout[stdout.index(marker) + len(marker):].strip()
    return stdout


def _clean_stderr(stderr: str) -> str:
    """Filter pip notice"""
    if not stderr:
        return ""
    lines = stderr.splitlines()
    kept = [
        line for line in lines
        if not any(p.match(line) for p in _PIP_WARNING_PATTERNS)
    ]
    return "\n".join(kept)

def _put_text_file(container, remote_dir: str, filename: str, text: str) -> str:
    """Upload `text` ass file inside the container via put_archive"""
    data = text.encode("utf-8")
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode="w") as tar:
        info = tarfile.TarInfo(name=filename)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    tar_stream.seek(0)
    container.put_archive(remote_dir, tar_stream)
    return f"{remote_dir.rstrip('/')}/{filename}"


def execute_python_code_in_sandbox(code: str, timeout: int = 60) -> dict:
    """
    Executes the given Python `code` string in a throwaway Docker container to help mitigate malicious commands.

    Returns:
      - dict with { "stdout": ..., "stderr": ..., "returncode": ... }
      or { "error": ... } in case of any timeout or unexpected exceptions.
    """
    try:
        client = docker.from_env()
    except DockerException as e:
        return {"error": f"Docker client unavailable: {e}"}

    unique_id = uuid.uuid4().hex
    container = None

    try:
        try:
            client.images.get(_DOCKER_IMAGE)
        except ImageNotFound:
            client.images.pull(_DOCKER_IMAGE)

        # Sleepy Container: the real code arrives later, via put_archive + exec_run
        container = client.containers.run(
            _DOCKER_IMAGE,
            ["sh", "-c", f"sleep {timeout + 10}"],
            name=f"datastorm_sandbox_{unique_id}",
            network_disabled=True,
            mem_limit="512m",
            detach=True,
        )

        script_path = _put_text_file(container, "/tmp", f"script_{unique_id}.py", code)

        exec_result = container.exec_run(
            cmd=["sh", "-c", f"{_INSTALL_CMD} && python3 {script_path}"],
            stdout=True,
            stderr=True,
            demux=True,
        )
        stdout_raw, stderr_raw = exec_result.output
        stdout_raw = (stdout_raw or b"").decode("utf-8", errors="replace")
        stderr_raw = (stderr_raw or b"").decode("utf-8", errors="replace")

        return {
            "stdout": _clean_stdout(stdout_raw),
            "stderr": _clean_stderr(stderr_raw),
            "returncode": exec_result.exit_code,
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        if container is not None:
            try:
                container.remove(force=True)
            except Exception:
                pass  # cleanup best-effort


def execute_python_script(python_script: str) -> dict:
    """
    A function that simulates the node's logic in your pipeline.
    Returns a dictionary with "stdout", "stderr", "returncode", or "error".
    """
    result = execute_python_code_in_sandbox(python_script, timeout=60)
    return result


if __name__ == "__main__":
    code_to_run = r'''
        # Question: Is there a significant correlation between the priority level and resolution time of incidents across all categories?
        
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        from scipy.stats import spearmanr
        
        # Load the CSV file into a DataFrame
        file_path = os.path.join(SQL_RESULTS_DIR, "example.csv")
        df = pd.read_csv(file_path)
        
        # Assuming the CSV contains a 'resolution_time' column (in hours, days, etc.)
        # and 'priority' column is categorical, we need to encode priority levels numerically.
        priority_mapping = {
            "1 - Critical": 1,
            "2 - High": 2,
            "3 - Moderate": 3,
            "4 - Low": 4
        }
        df['priority_numeric'] = df['priority'].map(priority_mapping)
        
        # Check for missing values in relevant columns
        if df[['priority_numeric', 'resolution_time']].isnull().any().any():
            df = df.dropna(subset=['priority_numeric', 'resolution_time'])
        
        # Calculate the Spearman correlation between priority and resolution time
        correlation, p_value = spearmanr(df['priority_numeric'], df['resolution_time'])
        
        # Print the correlation result
        print(f"Spearman Correlation: {correlation}")
        print(f"P-value: {p_value}")
        
        # Visualize the relationship using a scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='priority_numeric', y='resolution_time', data=df, alpha=0.6)
        plt.title('Priority Level vs Resolution Time')
        plt.xlabel('Priority Level (Numeric)')
        plt.ylabel('Resolution Time')
        plt.xticks(ticks=[1, 2, 3, 4], labels=["1 - Critical", "2 - High", "3 - Moderate", "4 - Low"])
        plt.grid(True)
        plt.show()
        '''

    # Call our function
    print("=== Testing execute_python_script ===")
    result_dict = execute_python_script(code_to_run)
    
    # Display the results
    print("Result dictionary:", json.dumps(result_dict, indent=2))
    
    # If you want just stdout
    stdout_str = result_dict.get("stdout", "")
    print("\n=== STDOUT ===\n", stdout_str)
    stderr_str = result_dict.get("stderr", "")
    print("\n=== STDERR ===\n", stderr_str)
    
    # If there was an error, show it
    if "error" in result_dict:
        print("\n=== ERROR ===\n", result_dict["error"])