import sys

try:
    from gui_interface import TomographyGUI
except ImportError:
    sys.exit(1)

def check_dependencies():
    required_packages = [
        ('SimpleITK', 'SimpleITK'),
        ('VTK', 'vtk'),
        ('PySimpleGUI', 'PySimpleGUI'),
        ('NumPy', 'numpy'),
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        return False
    
    return True


def main():
    if not check_dependencies():
        return 1
    
    try:
        app = TomographyGUI()
        app.run()
        return 0
        
    except KeyboardInterrupt:
        return 0
        
    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())