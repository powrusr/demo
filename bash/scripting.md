# functions

## arguments

- $0 is reserved for the function's name
- $# holds no of parameters passed to function
- $* and $@ variables hold all parameters passed to function
    - "$* expands to  "$1 $2 $n"
    - "$@" expands to "$1" "$2" "$n"
    - not quoted they are the same

## returning values

- $? captures exit status function

```bash
my_function () {
  echo "cool beans"
  return 33
}

my_function
echo $?
```
```output
cool beans
33
```

```bash
cool_function () {
  local result="cool beans"
  echo "$result"  # or use printf
}

cool_result="$(cool_function)"
echo $cool_result
```
```output
cool beans
```

