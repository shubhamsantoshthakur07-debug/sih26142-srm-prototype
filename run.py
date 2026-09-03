"""
One-click Runner for SIH26142 - Super Resolution Mapping (SRM) Console.
Supports both local Windows execution and Cloud environments (Render, Railway, Heroku).
"""

import sys
import os
import uvicorn

if __name__ == "__main__":
    # Ensure current directory is in sys.path
    project_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_dir)

    # Read port from environment (Render passes $PORT e.g. 10000)
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"  # Required for cloud hosts like Render to route public traffic

    print("=" * 65, flush=True)
    print("  SIH26142: DEEP LEARNING SUPER RESOLUTION MAPPING (SRM)", flush=True)
    print("  National Technical Research Organisation (NTRO) Console", flush=True)
    print("=" * 65, flush=True)
    print(f"  Binding Host: {host}", flush=True)
    print(f"  Listening Port: {port}", flush=True)
    print(f"  Console URL: http://localhost:{port}", flush=True)
    print("=" * 65, flush=True)

    uvicorn.run("server.app:app", host=host, port=port, log_level="info")
