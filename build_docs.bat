@echo OFF
echo "Building documentation..."
cd docs
sphinx-build -b html . _build
echo "Build complete. Open docs/_build/index.html to view the documentation."
cd .. 