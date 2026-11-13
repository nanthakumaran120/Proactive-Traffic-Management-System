import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
import websockets

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
CDP_PORT = 9222
ST_PORT = 8503
OUT_DIR = Path("screenshots")
ARTIFACT_DIR = Path(r"C:\Users\DELL\.gemini\antigravity\brain\ba05548c-af3d-4481-92f8-4078d3b44b3a")

OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


async def send_cdp(ws, method, params=None, msg_id=1):
    payload = {"id": msg_id, "method": method, "params": params or {}}
    await ws.send(json.dumps(payload))
    while True:
        resp = json.loads(await ws.recv())
        if resp.get("id") == msg_id:
            return resp.get("result", {})


async def capture_page_cdp(url: str, out_path: Path):
    # Get tabs from Chrome
    with urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json") as resp:
        tabs = json.loads(resp.read().decode())
    
    ws_url = tabs[0]["webSocketDebuggerUrl"]
    async with websockets.connect(ws_url, max_size=25*1024*1024) as ws:
        await send_cdp(ws, "Page.enable", msg_id=1)
        await send_cdp(ws, "Emulation.setDeviceMetricsOverride", {
            "width": 1920,
            "height": 1080,
            "deviceScaleFactor": 1,
            "mobile": False,
        }, msg_id=2)
        
        print(f"Navigating to {url}...")
        await send_cdp(ws, "Page.navigate", {"url": url}, msg_id=3)
        
        # Wait 5 seconds for Streamlit WebSocket to connect and render
        await asyncio.sleep(5.0)
        
        print("Capturing screenshot...")
        res = await send_cdp(ws, "Page.captureScreenshot", {"format": "png"}, msg_id=4)
        img_bytes = base64.b64decode(res["data"])
        out_path.write_bytes(img_bytes)
        print(f"Saved {out_path} ({len(img_bytes)} bytes)")
        dest = ARTIFACT_DIR / out_path.name
        shutil.copy2(out_path, dest)
        print(f"Copied to {dest}")


def wait_for_st(port=ST_PORT, timeout=20.0):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def wait_for_chrome(port=CDP_PORT, timeout=10.0):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=1) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


async def run_all():
    # Start Chrome in background with remote debugging
    chrome_proc = subprocess.Popen([
        CHROME_PATH,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        f"--remote-debugging-port={CDP_PORT}",
        "--window-size=1920,1080",
        "about:blank",
    ])
    try:
        if not wait_for_chrome():
            print("Chrome CDP failed to start!")
            return
        
        # 1. Run User Views
        env_user = os.environ.copy()
        env_user["TRAFFIC_DEV_ROLE"] = "user"
        st_proc = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless", "true",
            "--server.port", str(ST_PORT),
            "--server.address", "127.0.0.1",
        ], env=env_user)
        try:
            if wait_for_st():
                await asyncio.sleep(2.0)
                await capture_page_cdp(f"http://127.0.0.1:{ST_PORT}/?page=Home", OUT_DIR / "user_home.png")
                await capture_page_cdp(f"http://127.0.0.1:{ST_PORT}/?page=Find+Route", OUT_DIR / "user_find_route.png")
        finally:
            st_proc.terminate()
            try:
                st_proc.wait(timeout=5)
            except Exception:
                st_proc.kill()
            await asyncio.sleep(1.0)
            
        # 2. Run Admin Views
        env_admin = os.environ.copy()
        env_admin["TRAFFIC_DEV_ROLE"] = "admin"
        st_proc = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless", "true",
            "--server.port", str(ST_PORT),
            "--server.address", "127.0.0.1",
        ], env=env_admin)
        try:
            if wait_for_st():
                await asyncio.sleep(2.0)
                await capture_page_cdp(f"http://127.0.0.1:{ST_PORT}/?page=Overview", OUT_DIR / "admin_overview.png")
                await capture_page_cdp(f"http://127.0.0.1:{ST_PORT}/?page=Video+Analytics", OUT_DIR / "admin_video_analytics.png")
                await capture_page_cdp(f"http://127.0.0.1:{ST_PORT}/?page=Emergency+Detection", OUT_DIR / "admin_emergency_detection.png")
        finally:
            st_proc.terminate()
            try:
                st_proc.wait(timeout=5)
            except Exception:
                st_proc.kill()
            await asyncio.sleep(1.0)
            
    finally:
        chrome_proc.terminate()
        try:
            chrome_proc.wait(timeout=5)
        except Exception:
            chrome_proc.kill()


if __name__ == "__main__":
    asyncio.run(run_all())

