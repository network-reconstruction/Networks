import os
import glob
import ast
import sys
from typing import List, Set

# List of common standard library modules (incomplete, but covers many cases)
STANDARD_LIBS = {
    'abc', 'aifc', 'argparse', 'array', 'ast', 'asynchat', 'asyncio', 'asyncore', 'atexit',
    'audioop', 'base64', 'bdb', 'binascii', 'binhex', 'bisect', 'builtins', 'bz2', 'calendar',
    'cgi', 'cgitb', 'chunk', 'cmath', 'cmd', 'code', 'codecs', 'collections', 'colorsys',
    'compileall', 'concurrent', 'configparser', 'contextlib', 'contextvars', 'copy', 'copyreg',
    'crypt', 'csv', 'ctypes', 'curses', 'dataclasses', 'datetime', 'dbm', 'decimal', 'difflib',
    'dis', 'distutils', 'doctest', 'dummy_threading', 'email', 'encodings', 'ensurepip', 'enum',
    'errno', 'faulthandler', 'fcntl', 'filecmp', 'fileinput', 'fnmatch', 'formatter', 'fractions',
    'ftplib', 'functools', 'gc', 'getopt', 'getpass', 'gettext', 'glob', 'grp', 'gzip', 'hashlib',
    'heapq', 'hmac', 'html', 'http', 'imaplib', 'imghdr', 'imp', 'importlib', 'inspect', 'io',
    'ipaddress', 'itertools', 'json', 'keyword', 'lib2to3', 'linecache', 'locale', 'logging',
    'lzma', 'mailbox', 'mailcap', 'mimetypes', 'mmap', 'modulefinder', 'msilib', 'multiprocessing',
    'netrc', 'nis', 'nntplib', 'numbers', 'operator', 'optparse', 'os', 'parser', 'pathlib',
    'pdb', 'pickle', 'pickletools', 'pipes', 'pkgutil', 'platform', 'plistlib', 'poplib', 'posix',
    'pprint', 'profile', 'pstats', 'pty', 'pwd', 'py_compile', 'pyclbr', 'pydoc', 'queue', 'quopri',
    'random', 're', 'readline', 'resource', 'rlcompleter', 'runpy', 'sched', 'secrets', 'select',
    'selectors', 'shelve', 'shlex', 'shutil', 'signal', 'site', 'smtpd', 'smtplib', 'sndhdr', 'socket',
    'socketserver', 'spwd', 'sqlite3', 'ssl', 'stat', 'statistics', 'string', 'stringprep', 'struct',
    'subprocess', 'sunau', 'symbol', 'symtable', 'sys', 'sysconfig', 'syslog', 'tabnanny', 'tarfile',
    'telnetlib', 'tempfile', 'termios', 'test', 'textwrap', 'threading', 'time', 'timeit', 'tkinter',
    'token', 'tokenize', 'trace', 'traceback', 'tracemalloc', 'tty', 'turtle', 'types', 'typing',
    'unicodedata', 'unittest', 'urllib', 'uu', 'uuid', 'venv', 'warnings', 'wave', 'weakref', 'webbrowser',
    'wsgiref', 'xdrlib', 'xml', 'xmlrpc', 'zipapp', 'zipfile', 'zipimport', 'zlib'
}

def find_python_files(root_dir: str) -> List[str]:
    """Recursively find all Python files in the specified root directory."""
    return glob.glob(os.path.join(root_dir, '**', '*.py'), recursive=True)

def extract_imports(file_path: str) -> Set[str]:
    """Extract import statements from a Python file to identify dependencies."""
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read(), filename=file_path)
    
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])
    
    return imports

def gather_dependencies(root_dir: str) -> Set[str]:
    """Gather all unique dependencies from Python files in the root directory."""
    python_files = find_python_files(root_dir)
    dependencies = set()
    for file_path in python_files:
        dependencies.update(extract_imports(file_path))
    
    # Filter out standard library modules
    dependencies = {dep for dep in dependencies if dep not in STANDARD_LIBS}
    
    return dependencies

def create_requirements_txt(dependencies: Set[str]):
    """Create a requirements.txt file with the specified dependencies."""
    with open('requirements.txt', 'w') as req_file:
        for dependency in dependencies:
            req_file.write(f"{dependency}\n")

def main():
    #root directory is argument of the script
    root_dir = sys.argv[1]
    dependencies = gather_dependencies(root_dir)
    create_requirements_txt(dependencies)
    print("requirements.txt file created successfully.")
    print("Please review the file and remove any unnecessary dependencies.")
    print("After reviewing, you can install the dependencies using:\npip install -r requirements.txt")


if __name__ == "__main__":
    main()
