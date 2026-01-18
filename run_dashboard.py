import os
import socket
import subprocess
import sys
import time
import webbrowser


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(here, "streamlit_app.py")
    if not os.path.exists(app_path):
        print(f"Erreur: introuvable: {app_path}")
        return 2

    port = _pick_free_port()
    url = f"http://localhost:{port}"

    # Lance Streamlit dans ce même environnement Python.
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.address",
        "localhost",
        "--server.port",
        str(port),
    ]

    print("Lancement du dashboard…")
    print("URL:", url)

    # Démarre Streamlit et ouvre le navigateur après un court délai.
    proc = subprocess.Popen(cmd, cwd=here)
    time.sleep(1.2)
    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        return int(proc.wait())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
