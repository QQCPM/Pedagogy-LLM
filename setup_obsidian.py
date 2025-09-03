#!/usr/bin/env python3
"""
Setup script for Obsidian integration
Helps you configure the vault path
"""
import os
from pathlib import Path

def find_obsidian_vaults():
    """Try to automatically find Obsidian vaults"""
    common_locations = [
        Path.home() / "Documents" / "ObsidianVault",
        Path.home() / "Documents" / "Obsidian Vault", 
        Path.home() / "Obsidian",
        Path.home() / "Documents" / "Obsidian",
        Path.home() / "iCloud Drive (Archive)" / "Obsidian",
        Path.home() / "Library" / "Mobile Documents" / "iCloud~md~obsidian" / "Documents",
    ]
    
    found_vaults = []
    for location in common_locations:
        if location.exists():
            # Check if it looks like an Obsidian vault (has .obsidian folder)
            if (location / ".obsidian").exists():
                found_vaults.append(location)
            # Also check subdirectories for vaults
            try:
                for subdir in location.iterdir():
                    if subdir.is_dir() and (subdir / ".obsidian").exists():
                        found_vaults.append(subdir)
            except PermissionError:
                continue
    
    return found_vaults

def update_vault_path(vault_path):
    """Update the OBSIDIAN_VAULT path in ask.py"""
    ask_py_path = Path(__file__).parent / "ask.py"
    
    if not ask_py_path.exists():
        print("❌ ask.py not found")
        return False
    
    # Read current content
    with open(ask_py_path, 'r') as f:
        content = f.read()
    
    # Replace the vault path line
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith('OBSIDIAN_VAULT = '):
            lines[i] = f'OBSIDIAN_VAULT = Path("{vault_path}")'
            break
    
    # Write back
    with open(ask_py_path, 'w') as f:
        f.write('\n'.join(lines))
    
    return True

def main():
    print("🔧 OBSIDIAN VAULT SETUP")
    print("="*50)
    
    # Try to find vaults automatically
    found_vaults = find_obsidian_vaults()
    
    if found_vaults:
        print("📁 Found potential Obsidian vaults:")
        for i, vault in enumerate(found_vaults, 1):
            print(f"  {i}. {vault}")
        print(f"  {len(found_vaults) + 1}. Enter custom path")
        
        choice = input(f"\nChoose vault (1-{len(found_vaults) + 1}): ").strip()
        
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(found_vaults):
                vault_path = found_vaults[choice_idx]
            elif choice_idx == len(found_vaults):
                vault_path = Path(input("Enter vault path: ").strip())
            else:
                print("❌ Invalid choice")
                return
        except ValueError:
            print("❌ Invalid choice")
            return
    else:
        print("📁 No Obsidian vaults found automatically")
        vault_path = Path(input("Enter your Obsidian vault path: ").strip())
    
    # Validate path
    if not vault_path.exists():
        print(f"❌ Path doesn't exist: {vault_path}")
        create = input("Create this directory? (y/n): ").lower().strip()
        if create in ['y', 'yes']:
            vault_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {vault_path}")
        else:
            return
    
    # Update ask.py
    if update_vault_path(vault_path):
        print(f"✅ Updated vault path in ask.py")
        print(f"📁 Vault: {vault_path}")
        
        # Create basic folder structure
        folders = ["Mathematics", "AI-ML", "Physics", "Computer-Science", "General"]
        for folder in folders:
            (vault_path / folder).mkdir(exist_ok=True)
        
        print("✅ Created folder structure in vault")
        print("\n🚀 Ready to use!")
        print("Test with: python ask.py 'What are eigenvalues?'")
    else:
        print("❌ Failed to update ask.py")

if __name__ == "__main__":
    main()