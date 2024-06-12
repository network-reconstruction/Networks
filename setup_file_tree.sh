#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 <package_name>"
    exit 1
}

# Check if a package name was provided as an argument
if [ -z "$1" ]; then
    echo "Error: No package name provided."
    usage
fi

PACKAGE_NAME=$1

# Create the directory structure
mkdir -p $PACKAGE_NAME/$PACKAGE_NAME
mkdir -p $PACKAGE_NAME/$PACKAGE_NAME/submodule1
mkdir -p $PACKAGE_NAME/$PACKAGE_NAME/submodule1/sub_submodule1
mkdir -p $PACKAGE_NAME/$PACKAGE_NAME/submodule2
mkdir -p $PACKAGE_NAME/$PACKAGE_NAME/utils
mkdir -p $PACKAGE_NAME/tests
mkdir -p $PACKAGE_NAME/tests/test_submodule1
mkdir -p $PACKAGE_NAME/tests/test_submodule1/test_sub_submodule1
mkdir -p $PACKAGE_NAME/tests/test_submodule2
mkdir -p $PACKAGE_NAME/tests/test_utils

# Create the necessary files
touch $PACKAGE_NAME/$PACKAGE_NAME/__init__.py
touch $PACKAGE_NAME/$PACKAGE_NAME/submodule1/__init__.py
touch $PACKAGE_NAME/$PACKAGE_NAME/submodule1/module1.py
touch $PACKAGE_NAME/$PACKAGE_NAME/submodule1/sub_submodule1/__init__.py
touch $PACKAGE_NAME/$PACKAGE_NAME/submodule1/sub_submodule1/sub_module1.py
touch $PACKAGE_NAME/$PACKAGE_NAME/submodule2/__init__.py
touch $PACKAGE_NAME/$PACKAGE_NAME/submodule2/module2.py
touch $PACKAGE_NAME/$PACKAGE_NAME/utils/__init__.py
touch $PACKAGE_NAME/$PACKAGE_NAME/utils/helpers.py

touch $PACKAGE_NAME/tests/__init__.py
touch $PACKAGE_NAME/tests/test_submodule1/__init__.py
touch $PACKAGE_NAME/tests/test_submodule1/test_module1.py
touch $PACKAGE_NAME/tests/test_submodule1/test_sub_submodule1/__init__.py
touch $PACKAGE_NAME/tests/test_submodule1/test_sub_submodule1/test_sub_module1.py
touch $PACKAGE_NAME/tests/test_submodule2/__init__.py
touch $PACKAGE_NAME/tests/test_submodule2/test_module2.py
touch $PACKAGE_NAME/tests/test_utils/__init__.py
touch $PACKAGE_NAME/tests/test_utils/test_helpers.py

# Create the README.md file
cat <<EOL > $PACKAGE_NAME/README.md
# $PACKAGE_NAME

Description of your package.
EOL

# Create the .gitignore file
cat <<EOL > $PACKAGE_NAME/.gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
# According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
# However, in case of collaboration, if having platform-specific dependencies or dependencies
# having no cross-platform support, pipenv may install dependencies that don't work, or not
# install all needed dependencies.
Pipfile.lock

# PEP 582; used by e.g. github.com/David-OConnor/pyflow
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyderworkspace

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/
EOL

echo "Python package file tree with nested submodules set up in $PACKAGE_NAME."
