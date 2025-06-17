# sitecustomize.py - Global Python customization for torch compatibility
"""
This module is automatically imported by Python on startup to ensure
the torch.classes.__path__ fix is applied before any modules import torch.
This prevents the Streamlit file watcher "path._path" error during hot reloads.

Place this file in your project root or site-packages directory.
"""

try:
    import torch
    import types
    
    # Ensure torch.classes exists and has empty __path__ to prevent Streamlit watcher issues
    if not hasattr(torch, "classes"):
        torch.classes = types.ModuleType("torch.classes")
    
    # Always reset __path__ to empty list to prevent file watcher conflicts
    torch.classes.__path__ = []
    
except ImportError:
    # torch not available - no action needed
    pass
except Exception as e:
    # Silently handle any other errors during initialization
    pass