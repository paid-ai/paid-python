from importlib import metadata

try:
    __version__ = metadata.version("paid-python")
except metadata.PackageNotFoundError:
    __version__ = "unknown"
