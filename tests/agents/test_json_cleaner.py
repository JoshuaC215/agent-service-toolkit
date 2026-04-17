import json

import pytest

from agents.utils import JSONCleaner


@pytest.mark.parametrize("return_dict", [False, True])
def test_clean_json_repairs_trailing_comma_and_slices_code_fences(return_dict):
    cleaner = JSONCleaner()
    raw = """Some preface
```json
{ "a": 1, }
```
and trailing text
"""
    out = cleaner.clean_json(raw, return_dict=return_dict)
    if return_dict:
        assert out == {"a": 1}
    else:
        # Parse output string and compare structure
        parsed = json.loads(out)
        assert parsed == {"a": 1}


def test_clean_json_removes_control_chars_and_normalizes_unicode():
    cleaner = JSONCleaner()
    # Cafe\u0301 is decomposed "é"; include control char \x00 as well
    raw = 'xxx { "name": "Cafe\\u0301\\x00" } yyy'
    out = cleaner.clean_json(raw, return_dict=True)
    # Should normalize to NFC: "Café" and remove the control char
    assert out == {"name": "Café"}


def test_clean_json_valid_json_passthrough_structure():
    cleaner = JSONCleaner()
    raw = '{ "k": [1, 2, {"x": "y"}] }'
    out = cleaner.clean_json(raw, return_dict=False)
    # Round-trip parse and compare structure
    assert json.loads(out) == {"k": [1, 2, {"x": "y"}]}


def test_clean_json_missing_braces_raises():
    cleaner = JSONCleaner()
    with pytest.raises(ValueError) as ei:
        cleaner.clean_json("no braces here", return_dict=True)
    assert "Missing '{' or '}'" in str(ei.value)
