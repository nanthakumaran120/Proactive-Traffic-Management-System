"""Automated script to capture high-resolution desktop screenshots of the 5 key views."""

import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 8503
BASE_URL = f"http://127.0.0.1:{PORT}"
OUT_DIR = Path("screenshots")
ARTIFACT_DIR = Path(r"C:\Users\DELL\.gemini\antigravity\brain\ba05548c-af3d-4481-92f8-4078d3b44b3a")

OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def wait_for_server(url: str, timeout: float = 25.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"{url}/_stcore/health", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def capture(url: str, out_file: Path) -> None:
    print(f"Capturing {url} -> {out_file.name}...")
    cmd = [
        CHROME_PATH,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--window-size=1920,1080",
        "--hide-scrollbars",
        "--virtual-time-budget=7000",
        "--run-all-compositor-stages-before-draw",
        f"--screenshot={out_file.resolve()}",
        url,
    ]
    subprocess.run(cmd, check=True)
    if out_file.exists():
        print(f"Saved {out_file} ({out_file.stat().st_size} bytes)")
        dest = ARTIFACT_DIR / out_file.name
        shutil.copy2(out_file, dest)
        print(f"Copied to artifact dir: {dest}")


def run_session(role: str, captures: list[tuple[str, str]]) -> None:
    env = os.environ.copy()
    env["TRAFFIC_DEV_ROLE"] = role
    print(f"\n--- Starting Streamlit server for role: {role} ---")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.headless",
            "true",
            "--server.port",
            str(PORT),
            "--server.address",
            "127.0.0.1",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        if not wait_for_server(BASE_URL):
            print("Server failed to start in time!")
            return
        time.sleep(2.0)
        for page, fname in captures:
            url = f"{BASE_URL}/?page={page}"
            capture(url, OUT_DIR / fname)
    finally:
        print(f"Terminating Streamlit server for role: {role}")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(1.0)


def main() -> None:
    # 1. User views
    run_session(
        "user",
        [
            ("Home", "user_home.png"),
            ("Find+Route", "user_find_route.png"),
        ],
    )

    # 2. Admin views
    run_session(
        "admin",
        [
            ("Overview", "admin_overview.png"),
            ("Video+Analytics", "admin_video_analytics.png"),
            ("Emergency+Detection", "admin_emergency_detection.png"),
        ],
    )
    print("\nAll screenshots captured successfully!")


if __name__ == "__main__":
    main()

