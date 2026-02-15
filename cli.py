import argparse
import os
import yaml
from rich.console import Console
from .core import BrainManager

console = Console()

DEFAULT_YAML = """
name: "Mahestra"
core:
  dim: 2048
physics:
  beta: 25.0
  dt: 0.5
  stiffness: 0.9
  decay: 0.01
training:
  alpha: 0.05
storage:
  path: "./brains"
"""

def init_project(args):
    if os.path.exists("brain.yaml"):
        console.print("[yellow]⚠️ File brain.yaml sudah ada.[/yellow]")
        return
    
    with open("brain.yaml", "w") as f:
        f.write(DEFAULT_YAML.strip())
    
    os.makedirs("data", exist_ok=True)
    with open("data/contoh.txt", "w") as f:
        f.write("User: Halo.\nMahestra: Salam. Saya adalah AI berbasis Natumpy.\n")
        
    console.print("[bold green]✅ Project Manest Diinisialisasi![/bold green]")
    console.print("   1. Edit [bold]brain.yaml[/bold] untuk konfigurasi.")
    console.print("   2. Masukkan file .txt ke folder [bold]data/[/bold].")
    console.print("   3. Jalankan [bold]manest train[/bold].")

def train_model(args):
    manager = BrainManager()
    manager.initialize()
    manager.train("data")

def chat_model(args):
    manager = BrainManager()
    manager.initialize()
    manager.chat()

def main():
    parser = argparse.ArgumentParser(description="Manest: The Natumpy AI Cockpit")
    subparsers = parser.add_subparsers(dest="command", help="Perintah")

    # Command: init
    subparsers.add_parser("init", help="Buat project baru")

    # Command: train
    subparsers.add_parser("train", help="Latih AI dengan data di folder data/")

    # Command: chat
    subparsers.add_parser("chat", help="Ngobrol dengan AI")

    args = parser.parse_args()

    if args.command == "init":
        init_project(args)
    elif args.command == "train":
        train_model(args)
    elif args.command == "chat":
        chat_model(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
del(args)
    elif args.command == "chat":
        chat_model(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
