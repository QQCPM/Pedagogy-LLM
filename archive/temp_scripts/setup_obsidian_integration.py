#!/usr/bin/env python3
"""
Setup Obsidian Integration for Model Evaluation Results
"""
import json
from pathlib import Path
from datetime import datetime

def find_obsidian_vaults():
    """Try to find Obsidian vaults on the system"""
    possible_locations = [
        Path.home() / "Documents/Obsidian",
        Path.home() / "Documents/ObsidianVault", 
        Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents",
        Path.home() / "Obsidian",
        Path.home() / "Desktop/Obsidian",
        Path.home() / "Downloads"
    ]
    
    found_vaults = []
    for location in possible_locations:
        if location.exists():
            # Look for .obsidian folders which indicate Obsidian vaults
            for item in location.iterdir():
                if item.is_dir() and (item / ".obsidian").exists():
                    found_vaults.append(item)
                    print(f"📁 Found Obsidian vault: {item}")
    
    return found_vaults

def create_evaluation_folder_structure(vault_path: Path):
    """Create folder structure for evaluation results in Obsidian vault"""
    
    # Create main evaluation folder
    eval_folder = vault_path / "Model Evaluations"
    eval_folder.mkdir(exist_ok=True)
    
    # Create subfolders
    subfolders = [
        "Raw Results",
        "Ground Rules Results", 
        "Comparative Analysis",
        "Performance Metrics"
    ]
    
    for subfolder in subfolders:
        (eval_folder / subfolder).mkdir(exist_ok=True)
        print(f"📂 Created folder: {eval_folder / subfolder}")
    
    return eval_folder

def update_obsidian_path_in_scripts(vault_path: Path):
    """Update the Obsidian vault path in relevant scripts"""
    
    # Update ask.py
    ask_py = Path("ask.py")
    if ask_py.exists():
        content = ask_py.read_text()
        old_line = 'OBSIDIAN_VAULT = Path("/Users/quangnguyen/Downloads/hello")'
        new_line = f'OBSIDIAN_VAULT = Path("{vault_path}")'
        
        if old_line in content:
            updated_content = content.replace(old_line, new_line)
            ask_py.write_text(updated_content)
            print(f"✅ Updated ask.py with vault path: {vault_path}")
        else:
            print(f"⚠️ Could not find old path in ask.py - please update manually")
    
    # Update generate_obsidian_report.py to use the vault path
    report_py = Path("generate_obsidian_report.py")
    if report_py.exists():
        content = report_py.read_text()
        # Add a function to save directly to Obsidian vault
        vault_save_function = f'''
def save_to_obsidian_vault(report_content: str, filename: str = None) -> str:
    """Save report directly to Obsidian vault"""
    vault_path = Path("{vault_path}")
    eval_folder = vault_path / "Model Evaluations" / "Comparative Analysis"
    eval_folder.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Model_Evaluation_Report_{timestamp}.md"
    
    output_file = eval_folder / filename
    output_file.write_text(report_content, encoding='utf-8')
    
    print(f"📝 Saved to Obsidian vault: {{output_file}}")
    return str(output_file)
'''
        
        # Insert the function before the main() function
        if "def main():" in content and "def save_to_obsidian_vault" not in content:
            content = content.replace("def main():", vault_save_function + "\ndef main():")
            report_py.write_text(content)
            print(f"✅ Added Obsidian vault integration to generate_obsidian_report.py")

def main():
    print("🔧 Setting up Obsidian Integration for Model Evaluations")
    print("=" * 60)
    
    # Try to find existing Obsidian vaults
    print("🔍 Searching for Obsidian vaults...")
    found_vaults = find_obsidian_vaults()
    
    if found_vaults:
        print(f"\n📁 Found {len(found_vaults)} Obsidian vault(s):")
        for i, vault in enumerate(found_vaults, 1):
            print(f"  {i}. {vault}")
        
        # Let user choose or provide custom path
        print(f"\nOptions:")
        for i, vault in enumerate(found_vaults, 1):
            print(f"  {i}. Use {vault}")
        print(f"  {len(found_vaults) + 1}. Enter custom path")
        
        try:
            choice = input(f"\nSelect vault (1-{len(found_vaults) + 1}): ").strip()
            
            if choice.isdigit() and 1 <= int(choice) <= len(found_vaults):
                selected_vault = found_vaults[int(choice) - 1]
            elif choice == str(len(found_vaults) + 1):
                custom_path = input("Enter full path to your Obsidian vault: ").strip()
                selected_vault = Path(custom_path)
                if not selected_vault.exists():
                    print(f"❌ Path does not exist: {selected_vault}")
                    return
            else:
                print("❌ Invalid choice")
                return
                
        except (ValueError, KeyboardInterrupt):
            print("❌ Invalid input or cancelled")
            return
    else:
        print("❌ No Obsidian vaults found automatically.")
        custom_path = input("Enter full path to your Obsidian vault: ").strip()
        if not custom_path:
            print("❌ No path provided")
            return
        selected_vault = Path(custom_path)
        if not selected_vault.exists():
            print(f"❌ Path does not exist: {selected_vault}")
            return
    
    print(f"\n🎯 Using Obsidian vault: {selected_vault}")
    
    # Create folder structure
    print("\n📂 Creating folder structure...")
    eval_folder = create_evaluation_folder_structure(selected_vault)
    
    # Update scripts
    print("\n🔧 Updating scripts...")
    update_obsidian_path_in_scripts(selected_vault)
    
    # Create a sample report to test
    sample_report = f"""# Model Evaluation Setup Complete

**Setup Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Vault Path:** {selected_vault}
**Evaluation Folder:** {eval_folder}

## Folder Structure Created

- **Raw Results:** Individual model outputs without educational formatting
- **Ground Rules Results:** Outputs using research-focused ground rules prompting  
- **Comparative Analysis:** Side-by-side comparisons and improvement metrics
- **Performance Metrics:** Speed, length, and quality statistics

## Next Steps

1. Run model evaluations: `python3 evaluate_new_models.py`
2. Generate reports: `python3 generate_obsidian_report.py results.json`
3. Reports will automatically save to this vault's Model Evaluations folder

---

*Generated by Educational LLM Evaluation Setup*
"""
    
    setup_file = eval_folder / "Setup_Complete.md"
    setup_file.write_text(sample_report)
    
    print(f"\n✅ Setup complete!")
    print(f"📝 Sample file created: {setup_file}")
    print(f"🎯 Evaluation results will be saved to: {eval_folder}")
    print(f"\nYour Obsidian vault is now ready for model evaluation results!")

if __name__ == "__main__":
    main()