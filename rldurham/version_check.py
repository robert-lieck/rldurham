import sys
import warnings
import requests
from packaging.version import parse
from packaging.specifiers import SpecifierSet
from importlib.metadata import version, PackageNotFoundError

PACKAGE_NAME = "rldurham"


def check_for_update():
    try:
        # Get the installed version using importlib.metadata
        try:
            installed_version = version(PACKAGE_NAME)
        except PackageNotFoundError:
            return  # Package not installed, skip update check

        # Fetch latest version data from PyPI
        response = requests.get(f"https://pypi.org/pypi/{PACKAGE_NAME}/json", timeout=3)
        response.raise_for_status()
        data = response.json()

        # Determine the latest compatible version
        latest_version = None
        for ver in sorted(data["releases"], reverse=True, key=parse):
            requires_python = data["releases"][ver][0].get("requires_python", "")
            if requires_python and not _is_python_version_compatible(requires_python):
                continue
            latest_version = ver
            break

        # Compare versions and notify if an update is available
        if latest_version and parse(installed_version) < parse(latest_version):
            warnings.warn(
                f"A newer version of {PACKAGE_NAME} is available ({latest_version}). "
                f"You have {installed_version}. Upgrade with:\n\n    pip install --upgrade {PACKAGE_NAME}\n",
                UserWarning
            )

    except requests.RequestException:
        pass  # Silently fail on network issues


def _is_python_version_compatible(requires_python: str) -> bool:
    """Check if the current Python version satisfies 'requires_python'."""
    if not requires_python:
        return True
    try:
        spec = SpecifierSet(requires_python)
        return spec.contains(".".join(map(str, sys.version_info[:3])))
    except Exception:
        return True  # Assume compatible if parsing fails


# Run check on import
check_for_update()
