import pytest
import warnings
import sys
from pathlib import Path

# Add project directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import projectFiles.OutputVerificationModule as verify

def test_check_match_uniqueness_valid():
    """No warning should be issued for different image IDs."""
    result = verify.check_match_uniqueness("img001", "img002", matches=[])
    assert result is True

def test_check_match_uniqueness_warns_for_same_ids():
    """A warning should be issued when image IDs are the same."""
    with pytest.warns(UserWarning, match="Non-unique image match detected"):
        result = verify.check_match_uniqueness("img001", "img001", matches=[])
        assert result is True

def test_check_match_uniqueness_warns_with_matches():
    """Still warns even if matches are present."""
    dummy_matches = [object()] * 5
    with pytest.warns(UserWarning):
        verify.check_match_uniqueness("imgA", "imgA", dummy_matches)
