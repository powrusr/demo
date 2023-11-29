# regex

## regex functions

|function|desciption|
|--|:--:|
|re.match| determine whether it matches at the **beginning** of a string|
|re.search|finds a match of a pattern **anywhere** in the string|
|re.findall|returns a **list** of all substrings that match a pattern|
|re.iter| same as re.findall but returns an **iterator**
|re.sub| replace all (or count) occurences of pattern in string
|re.split(delim, string)| split string into list using delimiter|

```python

import re

pattern = r"pam"
match = re.search(pattern, "eggspamsausage")
if match:
    print(match.group())  # pam -> returns string matched = pam
    print(match.start())  # 4   -> returns start of first match
    print(match.end())    # 7   -> returns end of first match
    print(match.span())   # (4, 7) -> returns start & end of first match as tuple
    print(match.string)   # eggspamsausage -> returns string passed into function


# re.sub
str = "My name is David. Hi David."
pattern = r"David"
new_str = re.sub(pattern, "James", str, count=1)
print(new_str)  # My name is James. Hi David.
```

## metacharacters

|char|description|example|
|:--:|-----------|-------|
|[]	|A set of characters	|"[a-m]"	|
|\	|Signals a special sequence (can also be used to escape special characters)	|"\d"	
|.	|Any character (except newline character)|	"he..o"	|
|^	|Starts with	|"^hello"	|
|$	|Ends with	|"world$"|	
|*	|Zero or more occurrences|	"aix*"	|
|+	|One or more occurrences	|"aix+"	|
|{}	|Exactly the specified number of occurrences|	"al{2}"	|
|\|	|Either or	|"falls|stays"	|
|()	|Capture and group|

## special sequences aka character classes

|Character|	Description|Example|
|---------|------------|-------|
\A	|Returns a match if the specified characters are at the beginning of the string|	"\AThe"	
\b	|Returns a match where the specified characters are at the beginning or at the end of a word	|r"\bain"
|||r"ain\b"	
\B	|Returns a match where the specified characters are present, but NOT at the beginning (or at the end) of a word|	r"\Bain"
|||r"ain\B"	
\d	|Returns a match where the string contains digits (numbers from 0-9)|	"\d"	
\D	|Returns a match where the string DOES NOT contain digits|	"\D"	
\s	|Returns a match where the string contains a white space character|	"\s"	
\S	|Returns a match where the string DOES NOT contain a white space character|	"\S"	
\w	|Returns a match where the string contains any word characters (characters from a to Z, digits from 0-9, and the underscore _ character)	|"\w"	
\W	|Returns a match where the string DOES NOT contain any word characters|	"\W"	
\Z	|Returns a match if the specified characters are at the end of the string	|"Spain\Z"

## sets

| set| description|
|--|---|
[ ] | Contains a set of characters to match
[amk] | Matches either a, m, or k. It does not match amk
[a-z] | Matches any alphabet from a to z
[a\\-z] | Matches a, -, or z. It matches - because \ escapes it
[a-] | Matches a or -, because - is not being used to indicate a series of characters
[-a] | As above, matches a or -
[a-z0-9] | Matches characters from a to z and also from 0 to 9
[(+*)] | Special characters become literal inside a set, so this matches (, +, *, and )
[^ab5] | Adding ^ excludes any character in the set. Here, it matches characters that are not a, b, or 5

## groups

|group|description|
|-----|-----------|
( ) | Matches the expression inside the parentheses and groups it
(? ) | Inside parentheses like this, ? acts as an extension notation. Its meaning depends on the character immediately to its right
(?PAB) | Matches the expression AB, and it can be accessed with the group name
(?aiLmsux) | Here, a, i, L, m, s, u, and x are flags:
||a — Matches ASCII only
||i — Ignore case
||L — Locale dependent
||m — Multi-line
||s — Matches all
||u — Matches unicode
||x — Verbose
(?:A) | Matches the expression as represented by A, but unlike (?PAB), it cannot be retrieved afterwards
(?#...) | A comment. Contents are for us to read, not for matching
A(?=B) | Lookahead assertion. This matches the expression A only if it is followed by B
A(?!B) | Negative lookahead assertion. This matches the expression A only if it is not followed by B
(?<=B)A | Positive lookbehind assertion. This matches the expression A only if B is immediately to its left. This can only matched fixed length expressions
(?<!B)A | Negative lookbehind assertion
(?P=name) | Matches the expression matched by an earlier group named “name”
(...)\\1 | The number 1 corresponds to the first group to be matched


```python

import re

pattern = r"(?P<first>abc)(?:def)(ghi)"
# named group matching 'abc' followed by non-capturing group 'def' followed by 'ghi'
match = re.match(pattern, "abcdefghi")

if match:
    print(match.group("first"))  # abc
    print(match.groups())  # ('abc', 'ghi')


"""
\A and \Z match the beginning and end of a string
\b matches the empty string between \w and \W characters, or \w characters and -- the beginning or end of the string. Informally, it represents the boundary between words
\B matches the empty string anywhere else
\b(cat)\b" basically matches the word "cat" surrounded by word boundaries
"""

str = "Please contact info@learn.com for assistance"
pattern = r"([\w\.-]+)@([\w\.-]+)(\.[\w\.]+)"

```

## sub pattern group

### with empty string

`result = re.sub(pattern,repl='')`

### Non-capturing group

Use non-capturing groups when you want to match something that is not just made up of characters (otherwise you'd just use a set of characters in square brackets) and you don't need to capture it

```python
# Replace "foo" when it's got non-words or line boundaries to the left and to the right
pattern = r'(?:\W|^)foo(?:\W|$)'
replacement = " FOO "

string = 'foo bar foo foofoo barfoobar foo'

re.sub(pattern,replacement,string)
# >>> ' FOO bar FOO foofoo barfoobar FOO '
```

## lookarounds

### lookbehind ?<= and ?<!

#### immediately preceeds

```python
# replace "bar" with "BAR" if it IS preceded by "foo"
pattern = "(?<=foo)bar"
replacement = "BAR"

string = "foo bar foobar"

re.sub(pattern,replacement,string)
# 'foo bar fooBAR'
```

#### NOT immediately preceeds

```python
# replace "bar" with BAR if it is NOT preceded by "foo"
pattern = "(?<!foo)bar"
replacement = "BAR"
string = "foo bar foobar"

re.sub(pattern, replacement, string)
# 'foo BAR foobar'
```

### lookahead ?= and ?!

#### immediately followed

```python
# replace "foo" only if it IS followed by "bar"
pattern = "foo(?=bar)"
replacement = "FOO"
string = "foo bar foobar"

re.sub(pattern,replacement,string)
# 'foo bar fooBAR'
```

#### NOT immediately followed

```python
# replace "foo" only if it is NOT followed by "bar"
pattern = "foo(?!bar)"
replacement = "FOO"
string = "foo bar foobar"

re.sub(pattern,replacement,string)
# 'FOO bar foobar'
```

## examples

```python

```
