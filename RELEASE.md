# PyPI Release Guide for GLiNER2

## Prerequisites

- [ ] Python 3.8+ installed
- [ ] PyPI account with API token configured
- [ ] Write access to the repository

## Release Steps

### 1. Update Version

Update version in `gliner2/__init__.py`:
```python
__version__ = "1.0.1"  # New version
```

### 2. Build Package

```bash
# Install build tools
pip install build twine

# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Build package
python -m build
```

### 3. Test Build (Optional)

```bash
# Test on TestPyPI first
twine upload --repository testpypi dist/*

# Install and test
pip install --index-url https://test.pypi.org/simple/ gliner2
```

### 4. Upload to PyPI

```bash
# Upload to production PyPI
twine upload dist/*
```

### 5. Create GitHub Release

1. Go to GitHub repository → Releases
2. Click "Create a new release"
3. Tag: `v1.0.1` (matching version)
4. Title: `GLiNER2 v1.0.1`
5. Description: Summary of changes
6. Attach built wheels from `dist/` folder

### 6. Verify Release

```bash
# Install from PyPI
pip install gliner2==1.0.1

# Test basic functionality
python -c "from gliner2 import GLiNER2; print('✓ Import successful')"
```

## Troubleshooting

- **Authentication error**: Configure PyPI token in `~/.pypirc` or use `--username __token__`
- **File exists error**: Version already exists on PyPI, increment version number
- **Build fails**: Check `pyproject.toml` dependencies and Python version compatibility

## Checklist

- [ ] Version updated in `__init__.py`
- [ ] Package builds without errors
- [ ] Uploaded to PyPI successfully
- [ ] GitHub release created
- [ ] Installation verified
