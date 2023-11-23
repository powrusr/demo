# pytest

## installation

[pytest docs](https://docs.pytest.org)

```bash
pip install -U pytest
pytest --version
```

## pytest.ini

```ini
[pytest]
addopts = --strict-markers
markers =
    slow: Run tests that use sample data from file (deselect with '-m "not slow"')
```

## skip & slow

```python
@pytest.mark.skip("work in process")
def test_some_new_feature():
    assert False # test should fail
```

```python
@pytest.mark.slow
def test_large_file(phonebook):
    with open("test_data.txt") as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            name = row["Name"]
            number = row["Phone Number"]
            phonebook.add(name, number)
    assert phonebook.is_consistent()
```

```python
@pytest.mark.skipif(sys.version_info < (3, 6)
reason=“requires python3.6 or higher”)
def test_phonebook_contains_names():
    phonebook = PhoneBook()
    assert 'Bob' in phonebook.names()
```


```bash
# run all tests except slow
python -m pytest "not slow"
```

## fixtures

[fixtures](https://docs.pytest.org/en/7.4.x/reference/fixtures.html#fixtures)

Fixture availability is determined from the perspective of the test. A fixture is only available for tests to request if they are in the scope that fixture is defined in.

```bash
pytest --fixtures
========================================= test session starts =========================================
platform linux -- Python 3.10.12, pytest-7.4.3, pluggy-1.3.0
rootdir: /home/duke/gh/docs
collected 0 items                                                                                     
cache -- .../_pytest/cacheprovider.py:532
    Return a cache object that can persist state between testing sessions.

capsysbinary -- .../_pytest/capture.py:1001
    Enable bytes capturing of writes to ``sys.stdout`` and ``sys.stderr``.

capfd -- .../_pytest/capture.py:1029
    Enable text capturing of writes to file descriptors ``1`` and ``2``.

capfdbinary -- .../_pytest/capture.py:1057
    Enable bytes capturing of writes to file descriptors ``1`` and ``2``.

capsys -- .../_pytest/capture.py:973
    Enable text capturing of writes to ``sys.stdout`` and ``sys.stderr``.

doctest_namespace [session scope] -- .../_pytest/doctest.py:757
    Fixture that returns a :py:class:`dict` that will be injected into the
    namespace of doctests.

pytestconfig [session scope] -- .../_pytest/fixtures.py:1353
    Session-scoped fixture that returns the session's :class:`pytest.Config`
    object.

record_property -- .../_pytest/junitxml.py:282
    Add extra properties to the calling test.

record_xml_attribute -- .../_pytest/junitxml.py:305
    Add extra xml attributes to the tag for the calling test.

record_testsuite_property [session scope] -- .../_pytest/junitxml.py:343
    Record a new ``<property>`` tag as child of the root ``<testsuite>``.

tmpdir_factory [session scope] -- .../_pytest/legacypath.py:302
    Return a :class:`pytest.TempdirFactory` instance for the test session.

tmpdir -- .../_pytest/legacypath.py:309
    Return a temporary directory path object which is unique to each test
    function invocation, created as a sub directory of the base temporary
    directory.

caplog -- .../_pytest/logging.py:570
    Access and control log capturing.

monkeypatch -- .../_pytest/monkeypatch.py:30
    A convenient fixture for monkey-patching.

recwarn -- .../_pytest/recwarn.py:30
    Return a :class:`WarningsRecorder` instance that records all warnings emitted by test functions.

tmp_path_factory [session scope] -- .../_pytest/tmpdir.py:245
    Return a :class:`pytest.TempPathFactory` instance for the test session.

tmp_path -- .../_pytest/tmpdir.py:260
    Return a temporary directory path object which is unique to each test
    function invocation, created as a sub directory of the base temporary
    directory.
```

```python
@pytest.fixture
def phonebook():
return PhoneBook()

# using phonebook fixture
def test_lookup_by_name(phonebook):
phonebook.add("Bob", "12345")
assert "12345" == phonebook.lookup(“Bob")

# using phonebook fixture
def test_missing_name_raises_error(phonebook):
with pytest.raises(KeyError):
phonebook.lookup("Bob")
```

### conftest.py

share fixtures across multiple files

```python
"""Shared fixtures"""

import pytest

from phonebook.phonenumbers import Phonebook


@pytest.fixture
def phonebook(tmpdir):
    """Provides an empty Phonebook"""
    return Phonebook(tmpdir)
```

```python
# content of tests/conftest.py
import pytest

@pytest.fixture
def order():
    return []

@pytest.fixture
def top(order, innermost):
    order.append("top")
```



## assertions on expected exceptions

use pytest.raises() as context manager to write assertions on exceptions

```python
def test_zero_division():
    with pytest.raises(ZeroDivisionError):
        1 / 0

def test_missing_name_raises_error(phonebook):
    with pytest.raises(KeyError):
        phonebook.lookup("Bob")


def myfunc():
    raise ValueError("Exception 123 raised")


def test_match():
    with pytest.raises(ValueError, match=r".* 123 .*"):
        myfunc()
```

```python
@pytest.mark.xfail(raises=IndexError)
def test_f():
    f()
```
## parametrize

```python
@pytest.mark.parametrize("test_input,expected", [("3+5", 8), ("2+4", 6), ("6*9", 42)])
def test_eval(test_input, expected):
    assert eval(test_input) == expected


@pytest.mark.parametrize("n,expected", [(1, 2), (3, 4)])
class TestClass:
    def test_simple_case(self, n, expected):
        assert n + 1 == expected

    def test_weird_simple_case(self, n, expected):
        assert (n * 1) + 1 == expected




@pytest.mark.parametrize("entry1,entry2,is_consistent", [
    (("Bob", "12345"), ("Anna", "01234"), True),
    (("Bob", "12345"), ("Sue", "12345"), False),
    (("Bob", "12345"), ("Sue", "123"), False),
])
def test_is_consistent(phonebook, entry1, entry2, is_consistent):
    phonebook.add(*entry1)
    phonebook.add(*entry2)
    assert phonebook.is_consistent() == is_consistent
```

### parametrize all test in module

To parametrize all tests in a module, you can assign to the pytestmark global variable:

```python
import pytest

pytestmark = pytest.mark.parametrize("n,expected", [(1, 2), (3, 4)])

class TestClass:
    def test_simple_case(self, n, expected):
        assert n + 1 == expected

    def test_weird_simple_case(self, n, expected):
        assert (n * 1) + 1 == expected
```

### mark individual test instances within parametrize

```python
@pytest.mark.parametrize(
    "test_input,expected",
    [("3+5", 8), ("2+4", 6), pytest.param("6*9", 42, marks=pytest.mark.xfail)],
)
def test_eval(test_input, expected):
    assert eval(test_input) == expected
```

### stacking parametrize decorators

```python
@pytest.mark.parametrize("x", [0, 1])
@pytest.mark.parametrize("y", [2, 3])
def test_foo(x, y):
    pass
```
