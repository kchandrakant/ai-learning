#!/usr/bin/env python3
"""
Harness Engineering Learning Path - Environment Verification
============================================================

Run this script to verify your environment is properly configured
for the harness engineering learning path.

Usage:
    python verify_setup.py
"""

import sys
import os
from pathlib import Path


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_status(name: str, status: bool, details: str = "") -> None:
    """Print status of a check."""
    icon = "✓" if status else "✗"
    color_start = "\033[92m" if status else "\033[91m"
    color_end = "\033[0m"
    
    status_text = f"{color_start}{icon}{color_end} {name}"
    if details:
        status_text += f" - {details}"
    print(status_text)


def check_python_version() -> bool:
    """Check Python version is 3.10+."""
    version = sys.version_info
    is_valid = version.major == 3 and version.minor >= 10
    print_status(
        "Python Version",
        is_valid,
        f"{version.major}.{version.minor}.{version.micro} {'(OK)' if is_valid else '(Need 3.10+)'}"
    )
    return is_valid


def check_package(package_name: str, import_name: str = None) -> bool:
    """Check if a package is installed."""
    import_name = import_name or package_name
    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "installed")
        print_status(package_name, True, f"v{version}")
        return True
    except ImportError:
        print_status(package_name, False, "not installed")
        return False


def check_core_packages() -> dict:
    """Check core required packages."""
    print_header("Core Packages")
    
    results = {}
    
    # LLM Clients
    results["openai"] = check_package("openai")
    results["anthropic"] = check_package("anthropic")
    
    # Embeddings
    results["sentence-transformers"] = check_package(
        "sentence-transformers", "sentence_transformers"
    )
    results["scikit-learn"] = check_package("scikit-learn", "sklearn")
    results["numpy"] = check_package("numpy")
    
    # Validation
    results["pydantic"] = check_package("pydantic")
    results["jsonschema"] = check_package("jsonschema")
    
    # Testing
    results["pytest"] = check_package("pytest")
    results["pytest-asyncio"] = check_package("pytest-asyncio", "pytest_asyncio")
    
    return results


def check_optional_packages() -> dict:
    """Check optional packages."""
    print_header("Optional Packages")
    
    results = {}
    
    # Sandboxing
    results["docker"] = check_package("docker")
    
    # Observability
    results["structlog"] = check_package("structlog")
    results["rich"] = check_package("rich")
    
    # Utilities
    results["tenacity"] = check_package("tenacity")
    results["tiktoken"] = check_package("tiktoken")
    results["python-dotenv"] = check_package("python-dotenv", "dotenv")
    results["aiofiles"] = check_package("aiofiles")
    
    return results


def check_api_keys() -> dict:
    """Check for API keys in environment."""
    print_header("API Keys (Environment Variables)")
    
    results = {}
    
    # OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    has_openai = bool(openai_key) and openai_key != "your-key-here"
    print_status(
        "OPENAI_API_KEY",
        has_openai,
        "configured" if has_openai else "not set (optional)"
    )
    results["openai"] = has_openai
    
    # Anthropic
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    has_anthropic = bool(anthropic_key) and anthropic_key != "your-key-here"
    print_status(
        "ANTHROPIC_API_KEY",
        has_anthropic,
        "configured" if has_anthropic else "not set (optional)"
    )
    results["anthropic"] = has_anthropic
    
    return results


def check_docker() -> bool:
    """Check if Docker is available."""
    print_header("Docker (for Module 5: Sandboxing)")
    
    try:
        import docker
        client = docker.from_env()
        client.ping()
        print_status("Docker Daemon", True, "running")
        return True
    except ImportError:
        print_status("Docker SDK", False, "not installed")
        return False
    except Exception as e:
        print_status("Docker Daemon", False, f"not running ({type(e).__name__})")
        return False


def check_directory_structure() -> bool:
    """Check that all module directories exist."""
    print_header("Module Structure")
    
    expected_modules = [
        "00_foundations",
        "01_context_design",
        "02_tool_selection",
        "03_constraints",
        "04_evaluation",
        "05_production",
        "06_case_studies",
        "07_self_improving",
        "08_benchmarks",
        "demo",
        "evolutions",
    ]
    
    base_path = Path(__file__).parent
    all_exist = True
    
    for module in expected_modules:
        module_path = base_path / module
        readme_path = module_path / "README.md"
        
        exists = module_path.exists() and readme_path.exists()
        all_exist = all_exist and exists
        
        print_status(
            module,
            exists,
            "ready" if exists else "missing"
        )
    
    return all_exist


def print_summary(results: dict) -> None:
    """Print summary and recommendations."""
    print_header("Summary")
    
    core_ok = all(results.get("core", {}).values())
    optional_ok = all(results.get("optional", {}).values())
    structure_ok = results.get("structure", False)
    
    print(f"\nCore packages:     {'Ready' if core_ok else 'Missing packages'}")
    print(f"Optional packages: {'Ready' if optional_ok else 'Some missing'}")
    print(f"Module structure:  {'Ready' if structure_ok else 'Incomplete'}")
    
    # Recommendations
    print("\n" + "-"*60)
    
    if core_ok and structure_ok:
        print("\n🎉 Your environment is ready for the learning path!")
        print("\nNext steps:")
        print("  1. Open 00_foundations/README.md to start learning")
        print("  2. Work through modules in order")
        print("  3. Complete exercises in each module")
    else:
        print("\n⚠️  Some setup is needed:")
        
        if not core_ok:
            print("\n  Install missing packages:")
            print("    pip install -r requirements.txt")
        
        if not structure_ok:
            print("\n  Module directories are missing.")
            print("  Re-clone the repository or check the file structure.")
    
    # API key note
    if not results.get("api_keys", {}).get("openai") and not results.get("api_keys", {}).get("anthropic"):
        print("\n📝 Note: No API keys detected.")
        print("   API keys are optional for reading the modules,")
        print("   but required for hands-on exercises.")
        print("   Set OPENAI_API_KEY or ANTHROPIC_API_KEY when ready.")
    
    # Docker note
    if not results.get("docker", False):
        print("\n📝 Note: Docker is not running.")
        print("   Docker is optional but recommended for Module 5.")
        print("   Install Docker Desktop for sandboxing exercises.")


def main():
    """Run all verification checks."""
    print("\n" + "="*60)
    print("  Harness Engineering - Environment Verification")
    print("="*60)
    
    results = {}
    
    # Python version
    print_header("Python Environment")
    python_ok = check_python_version()
    results["python"] = python_ok
    
    if not python_ok:
        print("\n❌ Python 3.10+ is required. Please upgrade Python.")
        sys.exit(1)
    
    # Core packages
    results["core"] = check_core_packages()
    
    # Optional packages
    results["optional"] = check_optional_packages()
    
    # API keys
    results["api_keys"] = check_api_keys()
    
    # Docker
    results["docker"] = check_docker()
    
    # Directory structure
    results["structure"] = check_directory_structure()
    
    # Summary
    print_summary(results)
    
    # Return code
    core_ok = all(results.get("core", {}).values())
    structure_ok = results.get("structure", False)
    
    sys.exit(0 if (core_ok and structure_ok) else 1)


if __name__ == "__main__":
    main()
