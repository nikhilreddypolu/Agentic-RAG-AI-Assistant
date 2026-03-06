modules = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("langchain", "langchain"),
    ("langgraph", "langgraph"),
    ("sentence-transformers", "sentence_transformers"),
    ("faiss-cpu", "faiss"),
    ("pymupdf", "fitz"),
    ("python-dotenv", "dotenv"),
    ("google-generativeai", "google.generativeai"),
]

import importlib
import sys

print(f"Running import checks with Python {sys.executable}\n")
results = []
for pkg_name, mod_name in modules:
    try:
        importlib.import_module(mod_name)
        results.append((pkg_name, "OK"))
    except Exception as e:
        results.append((pkg_name, f"MISSING or import error: {e}"))

for pkg, status in results:
    print(f"{pkg}: {status}")

# small exit code for CI convenience
missing = [r for r in results if not r[1].startswith("OK")]
if missing:
    print('\nSome packages are missing or failed to import.')
    sys.exit(2)
else:
    print('\nAll checked packages imported successfully.')
    sys.exit(0)
