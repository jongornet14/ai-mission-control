import re
import os
from pathlib import Path


def universal_gym_step(env, action):
    """
    Universal function to handle both old and new Gym step API compatibility.

    Args:
        env: The gym environment
        action: The action to take

    Returns:
        tuple: Always returns (obs, reward, done, truncated, info) regardless of Gym version
    """
    step_result = env.step(action)

    if len(step_result) == 4:
        # Old Gym API (before v0.26): (obs, reward, done, info)
        obs, reward, done, info = step_result
        truncated = False
    elif len(step_result) == 5:
        # New Gym API (v0.26+): (obs, reward, terminated, truncated, info)
        obs, reward, done, truncated, info = step_result
    else:
        raise ValueError(
            f"Unexpected step result length: {len(step_result)}. "
            f"Expected 4 or 5 values, got {step_result}"
        )

    return obs, reward, done, truncated, info


def apply_universal_step_to_files():
    """
    Apply the universal gym step function to all relevant files in the project.
    """
    import re
    from pathlib import Path

    # Files to process
    files_to_fix = [
        "scripts/environments/gym_wrapper.py",
        "scripts/distributed_worker.py",
        "scripts/universal_rl.py",
    ]

    # Common patterns to find and replace
    step_patterns = [
        # Pattern 1: Standard 5-value unpacking
        (
            r"(\w+), (\w+), (\w+), (\w+), (\w+) = (.+)\.step\((.+)\)",
            r"\1, \2, \3, \4, \5 = universal_gym_step(\6, \7)",
        ),
        # Pattern 2: With asterisks (malformed)
        (
            r"(\w+), \*, (\w+), (\w+), \*, (\w+) = (.+)\.step\((.+)\)",
            r"\1, reward, \2, \3, \4 = universal_gym_step(\5, \6)",
        ),
        # Pattern 3: 4-value unpacking (old style)
        (
            r"(\w+), (\w+), (\w+), (\w+) = (.+)\.step\((.+)\)",
            r"\1, \2, \3, truncated, \4 = universal_gym_step(\5, \6)",
        ),
        # Pattern 4: self.env.step specific
        (
            r"(\w+), (\w+), (\w+), (\w+), (\w+) = self\.env\.step\((.+)\)",
            r"\1, \2, \3, \4, \5 = universal_gym_step(self.env, \6)",
        ),
    ]

    fixed_files = []

    for file_path in files_to_fix:
        file_obj = Path(file_path)

        if not file_obj.exists():
            print(f"⚠️  File not found: {file_path}")
            continue

        print(f"🔧 Processing {file_path}...")

        # Read file
        with open(file_obj, "r") as f:
            content = f.read()

        original_content = content
        changes_made = 0

        # Apply all patterns
        for pattern, replacement in step_patterns:
            matches = re.findall(pattern, content)
            if matches:
                content = re.sub(pattern, replacement, content)
                changes_made += len(matches)
                print(f"  ✅ Applied pattern to {len(matches)} occurrence(s)")

        # Add import at the top if changes were made
        if changes_made > 0:
            # Check if function is already imported/defined
            if "def universal_gym_step(" not in content:
                # Add the function definition at the top after imports
                import_section = []
                rest_of_file = []

                lines = content.split("\n")
                in_imports = True

                for line in lines:
                    if in_imports and (
                        line.startswith("import ")
                        or line.startswith("from ")
                        or line.strip() == ""
                        or line.strip().startswith("#")
                    ):
                        import_section.append(line)
                    else:
                        in_imports = False
                        rest_of_file.append(line)

                # Insert the function after imports
                function_def = '''

                                def universal_gym_step(env, action):
                                    """
                                    Universal function to handle both old and new Gym step API compatibility.
                                    
                                    Args:
                                        env: The gym environment
                                        action: The action to take
                                        
                                    Returns:
                                        tuple: Always returns (obs, reward, done, truncated, info) regardless of Gym version
                                    """
                                    step_result = env.step(action)
                                    
                                    if len(step_result) == 4:
                                        # Old Gym API (before v0.26): (obs, reward, done, info)
                                        obs, reward, done, info = step_result
                                        truncated = False
                                    elif len(step_result) == 5:
                                        # New Gym API (v0.26+): (obs, reward, terminated, truncated, info)
                                        obs, reward, done, truncated, info = step_result
                                    else:
                                        raise ValueError(f"Unexpected step result length: {len(step_result)}. "
                                                        f"Expected 4 or 5 values, got {step_result}")
                                    
                                    return obs, reward, done, truncated, info

                                '''

                content = (
                    "\n".join(import_section) + function_def + "\n".join(rest_of_file)
                )

            # Write back the modified file
            with open(file_obj, "w") as f:
                f.write(content)

            fixed_files.append(file_path)
            print(f"  ✅ Fixed {changes_made} step calls in {file_path}")
        else:
            print(f"  ℹ️  No step calls found to fix in {file_path}")

    return fixed_files


def create_standalone_compatibility_module():
    """
    Create a standalone module for gym compatibility that can be imported anywhere.
    """
    compat_file = Path("scripts/gym_compatibility.py")

    compat_content = '''"""
Gym Step API Compatibility Module

This module provides compatibility functions to handle different versions of OpenAI Gym
that return different numbers of values from env.step().

Usage:
    from gym_compatibility import universal_gym_step
    
    # Instead of:
    # obs, reward, done, truncated, info = env.step(action)
    
    # Use:
    obs, reward, done, truncated, info = universal_gym_step(env, action)
"""

def universal_gym_step(env, action):
    """
    Universal function to handle both old and new Gym step API compatibility.
    
    Handles:
    - Old Gym API (before v0.26): env.step() returns (obs, reward, done, info)
    - New Gym API (v0.26+): env.step() returns (obs, reward, terminated, truncated, info)
    
    Args:
        env: The gym environment
        action: The action to take
        
    Returns:
        tuple: Always returns (obs, reward, done, truncated, info) regardless of Gym version
        
    Raises:
        ValueError: If env.step() returns unexpected number of values
    """
    step_result = env.step(action)
    
    if len(step_result) == 4:
        # Old Gym API (before v0.26): (obs, reward, done, info)
        obs, reward, done, info = step_result
        truncated = False
    elif len(step_result) == 5:
        # New Gym API (v0.26+): (obs, reward, terminated, truncated, info)
        obs, reward, done, truncated, info = step_result
    else:
        raise ValueError(f"Unexpected step result length: {len(step_result)}. "
                        f"Expected 4 or 5 values, got {step_result}")
    
    return obs, reward, done, truncated, info


def check_gym_version():
    """
    Check which version of Gym is installed and what step API it uses.
    
    Returns:
        dict: Information about the gym version and API
    """
    try:
        import gym
        
        # Create a test environment to check step API
        env = gym.make('CartPole-v1')
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Handle new reset API
        
        action = env.action_space.sample()
        step_result = env.step(action)
        env.close()
        
        return {
            'gym_version': gym.__version__,
            'step_returns': len(step_result),
            'step_api': 'new' if len(step_result) == 5 else 'old',
            'compatible': True
        }
        
    except Exception as e:
        return {
            'gym_version': 'unknown',
            'step_returns': 'unknown', 
            'step_api': 'unknown',
            'compatible': False,
            'error': str(e)
        }


if __name__ == "__main__":
    # Test the compatibility function
    info = check_gym_version()
    print("Gym Compatibility Check:")
    print("========================")
    for key, value in info.items():
        print(f"{key}: {value}")
'''

    with open(compat_file, "w") as f:
        f.write(compat_content)

    print(f"✅ Created standalone compatibility module: {compat_file}")
    return compat_file


# Main execution function
def fix_all_gym_step_issues():
    """
    Complete fix for all Gym step compatibility issues in the project.
    """
    print("🚀 Starting comprehensive Gym step compatibility fix...")
    print("=" * 60)

    # Step 1: Create standalone module
    compat_file = create_standalone_compatibility_module()

    # Step 2: Fix all files
    fixed_files = apply_universal_step_to_files()

    # Step 3: Summary
    print("\n" + "=" * 60)
    print("✅ Gym Step Compatibility Fix Complete!")
    print("=" * 60)

    if fixed_files:
        print("📁 Fixed files:")
        for file_path in fixed_files:
            print(f"  ✅ {file_path}")
    else:
        print("⚠️  No files needed fixing (or no step calls found)")

    print(f"\n📦 Created compatibility module: {compat_file}")

    print("\n🧪 Test the fix by running:")
    print("  docker run -it --rm --gpus all \\")
    print("    -v $(pwd):/workspace/project \\")
    print("    -v $(pwd)/distributed_shared:/workspace/distributed_shared \\")
    print("    ai-mission-control:latest \\")
    print("    python scripts/distributed_worker.py \\")
    print("      --worker_id 0 \\")
    print("      --shared_dir /workspace/distributed_shared \\")
    print("      --env CartPole-v1 \\")
    print("      --max_episodes 3")

    return {
        "compat_module": str(compat_file),
        "fixed_files": fixed_files,
        "success": True,
    }


if __name__ == "__main__":
    # Run the complete fix
    result = fix_all_gym_step_issues()
    print(f"\nResult: {result}")
