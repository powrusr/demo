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

### capsys

```python
def test_print_fizzbuzz(capsys):
    print_fizzbuzz(3)
    captured_stdout = capsys.readouterr().out
    assert captured_stdout == "1\n2\nFizz\n"
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


@pytest.mark.parametrize(
    "dice,expected_score",
    [
        ([1, 1, 2, 2, 2], 8),
        ([5, 5, 6, 6, 6], 28),
    ]
)
def test_full_house(dice, expected_score):
    assert full_house(dice) == expected_score


@pytest.mark.parametrize(
    "dice,expected_categories", [
        ((1, 2, 3, 4, 5),
         [(15, 'large_straight'),(15, 'chance'),(14, 'small_straight'),(5, 'fives'), (4, 'fours'), (3, 'threes'), (2, 'twos'), (1, 'ones')]),
        ((2, 2, 3, 3, 3),
	[(13, 'full_house'), (13, 'chance'), (10, 'two_pairs'), (9, 'three_of_a_kind'), (9, 'threes'), (4, 'twos')]),
        ((6, 6, 6, 6, 6),
	[(50, 'yatzi'), (30, 'sixes'), (30, 'chance'), (24, 'four_of_a_kind'), (18, 'three_of_a_kind')]),
    ])
def test_scores_in_categories(dice, expected_categories):
    assert [(score, fun.__name__) for (score, fun) in scores_in_categories(dice)] == expected_categories
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

## test doubles

### stubs

- eg replace a remote sensor over network to measure tire pressure
- simple strict control, no behaviour or logic is replaced
- each test you setup needs a hard coded response

make sensor object optional in initializer, so test double can be passed
otherwise just call real sensor

```python
from sensor import Sensor


class Alarm:

    def __init__(self, sensor=None):
        self._low_pressure_threshold = 17.0
        self._high_pressure_threshold = 21.0
        self._sensor = sensor or Sensor()
        self._is_alarm_on = False

--

from unittest.mock import Mock

from alarm import Alarm
from sensor import Sensor


def test_alarm_is_off_by_default():
    alarm = Alarm()
    assert not alarm.is_alarm_on

# our stub
class StubSensor:
    def sample_pressure(self):
        return 17.0


def test_alarm_is_on_at_lower_threshold():
    alarm = Alarm(StubSensor())
    alarm.check()
    assert alarm.is_alarm_on


def test_alarm_is_on_at_higher_threshold():
    stub = Mock(Sensor)   # these 2 lines are same as StubSensor class
    stub.sample_pressure.return_value = 21.0
    alarm = Alarm(stub)
    alarm.check()
    assert alarm.is_alarm_on
```

### fakes

- has an implementation with logic & behaviour but is unsuitable for production:
  - replace file operations with a StringIO collaborator
  - replace real DB with in-memory database
  - replace webserver with light-weight web server

#### StringIO

eg. because we can't wait testing some really big file

```python
import io

from html_pages import HtmlPagesConverter


def test_convert_second_page():
    fake_file = io.StringIO("""\
page one
PAGE_BREAK
page two
PAGE_BREAK
page three
""")
    converter = HtmlPagesConverter(fake_file)
    converted_text = converter.get_html_page(1)
    assert converted_text == "page two<br />"
```

### dummies

- use of None or an empty collection, shouldn't be used

### spies

**spy records method calls it receives, so you can check correctness**
**spy raises error afterwards in the assert**
(mock will immediately raise one)

- makes assertions on what happens, it can fail, a stub won't
  1. assert a return value or an exception
  2. test State change (API query)
  3. check function/method got called

eg using `Mock.assert_has_calls(expected_calls)`

#### manual spy

```python
from unittest.mock import Mock, call

from discounts import DiscountManager
from model_objects import DiscountData, Product, User


class SpyNotifier:
    def __init__(self):
        self.notified_users = []

    def notify(self, user, message):
        # don't send any messages from the unit test, record which users are notified
        self.notified_users.append(user)


def test_discount_for_users_with_spy():
    notifier = SpyNotifier()
    discount_manager = DiscountManager(notifier)
    product = Product("headphones")
    product.discounts = []
    discount_details = DiscountData("10% off")
    users = [User("user1", [product]), User("user2", [product])]

    discount_manager.create_discount(product, discount_details, users)

    assert product.discounts == [discount_details]
    assert users[0] in notifier.notified_users
    assert users[1] in notifier.notified_users
```

#### using unittest.mock spy

```python
def test_discount_for_users_with_mocking_framework_spy():
    notifier = Mock()
    discount_manager = DiscountManager(notifier)
    product = Product("headphones")
    product.discounts = []
    discount_details = DiscountData("10% off")
    users = [User("user1", [product]), User("user2", [product])]

    discount_manager.create_discount(product, discount_details, users)

    assert product.discounts == [discount_details]
    expected_calls = [
        call(users[0], f"You may be interested in a discount on this product! {product.name}"),
        call(users[1], f"You may be interested in a discount on this product! {product.name}"),
    ]
    notifier.notify.assert_has_calls(expected_calls)
```

### mocks

**a mock fails earlier in the test than a spy**
- told in advance what methods will be called with what arguments
- will raise error right away if call expectations aren't met

```python
class MockNotifier:
    def __init__(self):
        self.expected_user_notifications = []

    def notify(self, user, message):
        # don't send any messages from the unit test, check all notifications are expected
        expected_user = self.expected_user_notifications.pop(0)
        if user != expected_user:
            raise RuntimeError(f"got notification message for unexpected user {user.name}, was expecting {expected_user.name} instead")

    def expect_notification_to(self, user):
        self.expected_user_notifications.append(user)


def test_discount_for_users_with_mock():
    notifier = MockNotifier()
    discount_manager = DiscountManager(notifier)
    product = Product("headphones")
    product.discounts = []
    discount_details = DiscountData("10% off")
    users = [User("user1", [product]), User("user2", [product])]
    notifier.expect_notification_to(users[0])
    notifier.expect_notification_to(users[1])

    discount_manager.create_discount(product, discount_details, users)

    assert product.discounts == [discount_details]
```

## approval tests

```bash
pip install approvaltests
pip install pytest-approvaltests
pytest --approvaltests-use-reporter='PythonNative'
```
```python
from approvaltests.approvals import verify


def test_simple():
    result = "Hello ApprovalTests"
    verify(result)
```

pytest runner configuration:

Additional Arguments: `--approvaltests-use-reporter'PythonNative'`

create approval files in subdir through config file
approvals_config.json

```json
{
  "subdirectory": "approved_files"
}
```

### verify string

```python

import approvaltests
import pytest

from model_objects import SpecialOfferType
from receipt_printer import ReceiptPrinter


def test_no_discount(teller, cart, toothbrush, apples):
    teller.add_special_offer(SpecialOfferType.TEN_PERCENT_DISCOUNT, toothbrush, 10.0)
    cart.add_item_quantity(apples, 2.5)

    receipt = teller.checks_out_articles_from(cart)

    approvaltests.verify(ReceiptPrinter().print_receipt(receipt))


def test_ten_percent_discount(teller, cart, toothbrush):
    teller.add_special_offer(SpecialOfferType.TEN_PERCENT_DISCOUNT, toothbrush, 10.0)
    cart.add_item_quantity(toothbrush, 2)

    receipt = teller.checks_out_articles_from(cart)

    approvaltests.verify(ReceiptPrinter().print_receipt(receipt))
```

### best covering pairs

`approvaltests.verify_best_covering_pairs()`

```python
import unittest

import approvaltests

from gilded_rose import Item, GildedRose


def print_item(item):
    return f"Item({item.name}, sell_in={item.sell_in}, quality={item.quality})"


def update_quality_printer(args, result):
    return f"{args} => {print_item(result)}\n"


def test_update_quality():
    names = ["foo", "Cottage Cheese", "Backstage tickets", "Magic Wand"]
    sell_ins = [-1, 0, 1, 6, 11]
    qualitys = [0, 1, 2, 48, 49, 50]

    approvaltests.verify_best_covering_pairs(
        update_quality_for_item,
        [
            names,
            sell_ins,
            qualitys
        ],
        formatter=update_quality_printer
    )


def update_quality_for_item(name, sell_in, quality):
    item = Item(name, sell_in, quality)
    items = [item]
    gilded_rose = GildedRose(items)
    gilded_rose.update_quality()
    return item
```

```
Testing an optimized 30/120 scenarios:

['foo', -1, 0] => Item(foo, sell_in=-2, quality=0)
['Cottage Cheese', 0, 0] => Item(Cottage Cheese, sell_in=-1, quality=2)
['Backstage tickets', 1, 0] => Item(Backstage tickets, sell_in=0, quality=3)
['Magic Wand', 6, 0] => Item(Magic Wand, sell_in=6, quality=0)
...
```
### more advanced combination approvals

[more combination approval code](https://github.com/approvals/ApprovalTests.Python/blob/main/approvaltests/combination_approvals.py)

## coverage

[coverage docs](https://coverage.readthedocs.io/)

```bash
coverage help
coverage help run
coverage run --help
```

- run – Run a Python program and collect execution data.
- combine – Combine together a number of data files.
- erase – Erase previously collected coverage data.
- report – Report coverage results.
- html – Produce annotated HTML listings with coverage results.
- xml – Produce an XML report with coverage results.
- json – Produce a JSON report with coverage results.
- lcov – Produce an LCOV report with coverage results.
- annotate – Annotate source files with coverage results.
- debug – Get diagnostic information.

```bash
python3 -m pip install coverage

coverage run -m pytest arg1 arg2 arg3

$ coverage report -m
Name                      Stmts   Miss  Cover   Missing
-------------------------------------------------------
my_program.py                20      4    80%   33-35, 39
my_other_module.py           56      6    89%   17-23
-------------------------------------------------------
TOTAL                        76     10    87%


# to view lines in html
$ coverage html


# to measure branch coverage:
coverage run --branch myprog.py
```

### exclude branches

```python
def only_one_choice(x):
    if x:
        blah1()
        blah2()
    else:  # pragma: no cover
        # x is always true.
        blah3()
```


### pytest runner cov plugin

[pytest-cov plugin docs](https://pytest-cov.readthedocs.io/en/latest/)


