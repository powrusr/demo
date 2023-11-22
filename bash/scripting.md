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

## examples

```bash
die () {
  echo >&2 "$@"
  exit 1
}

convert_to_mp4 () {
  vid="$1"
  base=${vid:0:-4}  # drop last 4 chars eg .avi
  [[ "$#" -eq 1 ]] || die "1 arg required, $# provided"
  [[ -f ./$vid ]] || echo "$1 video file doesnt exist"
  [[ -f ./$vid ]] && echo "$vid file exists :)"
  ffmpeg -i ./"$vid" -vcodec mpeg4 -acodec aac ./"$base".mp4
}
```
[param expansion](https://www.gnu.org/software/bash/manual/html_node/Shell-Parameter-Expansion.html)

# testing

## table

| **Feature**                                                                 | **new test** [[                                           | **old test** [                                                    | **Example**                                                                                                         |
| --------------------------------------------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| string comparison                                                           | \>                                                        | \\> [(\*)](https://mywiki.wooledge.org/BashFAQ/031#np)            | [[ a > b ]] || echo "a does not come after b"                                                                       |
|  | <                                                                           | \\< [(\*)](https://mywiki.wooledge.org/BashFAQ/031#np)    | [[ az < za ]] && echo "az comes before za"                        |
|  | \= (or \==)                                                                 | \=                                                        | [[ a = a ]] && echo "a equals a"                                  |
|  | !=                                                                          | !=                                                        | [[ a != b ]] && echo "a is not equal to b"                        |
| integer comparison                                                          | \-gt                                                      | \-gt                                                              | [[ 5 -gt 10 ]] || echo "5 is not bigger than 10"                                                                    |
|   | \-lt                                                                        | \-lt                                                      | [[ 8 -lt 9 ]] && echo "8 is less than 9"                          |
|   | \-ge                                                                        | \-ge                                                      | [[ 3 -ge 3 ]] && echo "3 is greater than or equal to 3"           |
|   |   # drop last 4 chars eg .avi\-le                                                                        | \-le                                                      | [[ 3 -le 8 ]] && echo "3 is less than or equal to 8"              |
| \-eq                                                                        | \-eq                                                      | [[ 5 -eq 05 ]] && echo "5 equals 05"                              |
|   | \-ne                                                                        | \-ne                                                      | [[ 6 -ne 20 ]] && echo "6 is not equal to 20"                     |
| co   |nditional evaluation                                                      | &&                                                        | \-a [(\*\*)](https://mywiki.wooledge.org/BashFAQ/031#np2)         | [[ -n $var && -f $var ]] && echo "$var is a file"                                                                   |
|   |  \|\|  | \-o [(\*\*)](https://mywiki.wooledge.org/BashFAQ/031#np2) | [[ -b $var || -c $var ]] && echo "$var is a device"               |
| expression grouping  | (...)  | \\( ... \\) [(\*\*)](https://mywiki.wooledge.org/BashFAQ/031#np2) | [[ $var = img\* && ($var = \*.png || $var = \*.jpg) ]] &&<br>echo "$var starts with img and ends with .jpg or .png" |
| Pattern matching   | \= (or \==)    | (not available) | [[ $name = a\* ]] || echo "name does not start with an 'a': $name"                                                  |
[param expansion](https://www.gnu.org/software/bash/manual/html_node/Shell-Parameter-Expansion.html)
| [RegularExpression](https://mywiki.wooledge.org/RegularExpression) matching | \=~ | (not available) | [[ $(date) =~ ^Fri\\ ...\\ 13 ]] && echo "It's Friday the 13th!"|

