from pathlib import Path
import json
import os

def test_file_operations():
    # Test paths
    base_path = Path(__file__).parent.parent.resolve()
    story_path = base_path / "content" / "stories"
    arc_path = story_path / "arcs"
    
    print("\nTesting file operations:")
    print(f"Base path: {base_path}")
    print(f"Story path: {story_path}")
    print(f"Arc path: {arc_path}")
    
    # Create directories
    arc_path.mkdir(parents=True, exist_ok=True)
    
    # Test file writing
    test_data = {
        "title": "Test Story",
        "content": "This is a test."
    }
    
    test_file = arc_path / "test_story_arc.json"
    
    try:
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=4)
        print(f"\nSuccessfully wrote test file: {test_file}")
        print(f"File exists: {test_file.exists()}")
        print(f"File size: {test_file.stat().st_size} bytes")
    except Exception as e:
        print(f"\nError writing test file: {e}")

if __name__ == "__main__":
    test_file_operations() 