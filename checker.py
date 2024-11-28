import requests
import json
from packaging import version
import concurrent.futures

def get_compatible_version(package_line):
    if not package_line.strip() or package_line.startswith('#'):
        return package_line
    
    try:
        package_name = package_line.split('==')[0].strip()
        url = f"https://pypi.org/pypi/{package_name}/json"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            info = data.get('info', {})
            classifiers = info.get('classifiers', [])
            
            # Check Python version classifiers
            python_versions = [c for c in classifiers if "Python :: 3.10" in c]
            
            if python_versions:
                # Package supports Python 3.10
                return package_line
            else:
                # Try to find an older version that might work
                releases = list(data['releases'].keys())
                releases.sort(key=lambda x: version.parse(x), reverse=True)
                
                for rel in releases[:5]:  # Check the 5 most recent versions
                    release_data = data['releases'][rel]
                    if release_data:
                        return f"{package_name}=={rel}"
                
                return f"# {package_line} # Warning: Compatibility uncertain"
        
        return f"# {package_line} # Failed to check"
    except Exception as e:
        return f"# {package_line} # Error: {str(e)}"

def main():
    with open('requirements.txt', 'r') as f:
        requirements = [line.strip() for line in f if line.strip()]
    
    print("Checking PyPI for Python 3.10 compatibility...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(get_compatible_version, requirements))
    
    # Special handling for PyTorch
    torch_versions = """
torch>=2.0.1
torchvision==0.15.2
torchaudio==2.0.2
    """.strip().split('\n')
    
    # Combine results
    final_requirements = []
    for line in results:
        if any(torch_pkg in line for torch_pkg in ['torch==', 'torchvision==', 'torchaudio==']):
            continue
        final_requirements.append(line)
    final_requirements.extend(torch_versions)
    
    with open('requirements_py310.txt', 'w') as f:
        f.write("\n".join(final_requirements))
    
    print("Done! Check requirements_py310.txt for updated versions.")

if __name__ == "__main__":
    main()