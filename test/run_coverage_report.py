import sys
import pytest

def main():
    return pytest.main([
        "test",  # Path to your tests
        "--cov=src/projectFiles",
        "--cov-report=term-missing",
        "--cov-report=html:test/coverage_html",  # Save HTML report under test/coverage_html
    ])

if __name__ == "__main__":
    sys.exit(main())
