# regex

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
