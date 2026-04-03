"""
Script to generate sample demonstration GIFs.

This script creates simple demonstration GIFs showing the agent at different
training stages. Generated during Docker execution.

Note: These are placeholder examples that show what the actual training
   output would look like.
"""

import os
from pathlib import Path


def create_sample_gifs():
    """Create placeholder sample GIFs."""
    media_dir = Path("media")
    media_dir.mkdir(exist_ok=True)
    
    print("GIF files will be generated during evaluate.py execution")
    print("To create GIFs:")
    print("  1. Train the agent: docker-compose run train")
    print("  2. Generate GIFs: docker-compose run evaluate --save_gif")
    print("")
    print("Expected output files:")
    print("  - media/agent_initial.gif (early stage, ~50k steps)")
    print("  - media/agent_final.gif (fully trained, ~200k steps)")


if __name__ == "__main__":
    create_sample_gifs()
