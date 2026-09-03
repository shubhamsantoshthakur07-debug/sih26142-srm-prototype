"""
One-click Runner for SIH26142 - Super Resolution Mapping (SRM) Console.
"""

import sys
import os
import uvicorn

if __name__ == "__main__":
    # Ensure current directory is in sys.path
    project_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_dir)

    print("=" * 65)
    print("  SIH26142: DEEP LEARNING SUPER RESOLUTION MAPPING (SRM)")
    print("  National Technical Research Organisation (NTRO) Prototype")
    print("=" * 65)
    print("  Console URL: http://localhost:8000")
    print("  API Docs:    http://localhost:8000/docs")
    print("  Press CTRL+C to terminate.")
    print("=" * 65)

    uvicorn.run("server.app:app", host="127.0.0.1", port=8000, log_level="info")
