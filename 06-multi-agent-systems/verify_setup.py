"""
Verify that your environment is set up correctly for the Multi-Agent Systems course.
"""

import sys

def check_python_version():
    """Check Python version >= 3.10"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (need 3.10+)")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    import_name = import_name or package_name
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name} ({version})")
        return True
    except ImportError:
        print(f"❌ {package_name} not installed")
        return False

def check_api_keys():
    """Check for API key environment variables"""
    import os
    
    keys = {
        'OPENAI_API_KEY': 'OpenAI',
        'ANTHROPIC_API_KEY': 'Anthropic (optional)',
    }
    
    print("\n📋 API Keys:")
    results = []
    for key, name in keys.items():
        if os.environ.get(key):
            print(f"  ✅ {name} API key found")
            results.append(True)
        else:
            optional = 'optional' in name.lower()
            symbol = "⚠️" if optional else "❌"
            print(f"  {symbol} {name} API key not set ({key})")
            results.append(optional)
    
    return all(results)

def main():
    print("=" * 60)
    print("MULTI-AGENT SYSTEMS COURSE - SETUP VERIFICATION")
    print("=" * 60)
    
    all_good = True
    
    # Python version
    print("\n📋 Python Version:")
    all_good &= check_python_version()
    
    # Core packages
    print("\n📋 Core LLM Packages:")
    all_good &= check_package('openai')
    all_good &= check_package('anthropic')
    
    # Agent Frameworks
    print("\n📋 Agent Frameworks:")
    all_good &= check_package('langgraph')
    all_good &= check_package('langchain')
    
    # Memory
    print("\n📋 Memory & Storage:")
    all_good &= check_package('chromadb')
    
    # Utilities
    print("\n📋 Utilities:")
    all_good &= check_package('pydantic')
    all_good &= check_package('tenacity')
    
    # API Keys
    all_good &= check_api_keys()
    
    # Summary
    print("\n" + "=" * 60)
    if all_good:
        print("✅ All checks passed! You're ready to start.")
    else:
        print("⚠️  Some checks failed. Install missing packages:")
        print("   pip install -r requirements.txt")
        print("\n   Set API keys in your environment or .env file")
    print("=" * 60)
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())
