# scripting

## functions

### arguments

* $0 is reserved for the function's name
* $# holds no of parameters passed to function
* $\* and $@ variables hold all parameters passed to function
  * "$\* expands to "$1 $2 $n"
  * "$@" expands to "$1" "$2" "$n"
  * not quoted they are the same

### returning values

* $? captures exit status function

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

### examples

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

[param expansion](https://www.gnu.org/software/bash/manual/html\_node/Shell-Parameter-Expansion.html)

## testing

### table

<table data-header-hidden data-full-width="true"><thead><tr><th width="215.50000000000003"></th><th width="123"></th><th width="150"></th><th></th></tr></thead><tbody><tr><td><strong>Feature</strong></td><td><strong>new test</strong> [[</td><td><strong>old test</strong> [</td><td><strong>Example</strong></td></tr><tr><td>string comparison</td><td>></td><td>\> <a href="https://mywiki.wooledge.org/BashFAQ/031#np">(*)</a></td><td>[[ a > b ]] </td></tr><tr><td></td><td>&#x3C;</td><td>\&#x3C; <a href="https://mywiki.wooledge.org/BashFAQ/031#np">(*)</a></td><td>[[ az &#x3C; za ]] &#x26;&#x26; echo "az comes before za"</td></tr><tr><td></td><td>= (or ==)</td><td>=</td><td>[[ a = a ]] &#x26;&#x26; echo "a equals a"</td></tr><tr><td></td><td>!=</td><td>!=</td><td>[[ a != b ]] &#x26;&#x26; echo "a is not equal to b"</td></tr><tr><td>integer comparison</td><td>-gt</td><td>-gt</td><td>[[ 5 -gt 10 ]] </td></tr><tr><td></td><td>-lt</td><td>-lt</td><td>[[ 8 -lt 9 ]] &#x26;&#x26; echo "8 is less than 9"</td></tr><tr><td></td><td>-ge</td><td>-ge</td><td>[[ 3 -ge 3 ]] &#x26;&#x26; echo "3 is greater than or equal to 3"</td></tr><tr><td></td><td>-le</td><td>-le</td><td>[[ 3 -le 8 ]] &#x26;&#x26; echo "3 is less than or equal to 8"</td></tr><tr><td></td><td>-eq</td><td>-eq</td><td>[[ 5 -eq 05 ]] &#x26;&#x26; echo "5 equals 05"</td></tr><tr><td></td><td>-ne</td><td>-ne</td><td>[[ 6 -ne 20 ]] &#x26;&#x26; echo "6 is not equal to 20"</td></tr><tr><td>conditional evaluation</td><td>&#x26;&#x26;</td><td>-a <a href="https://mywiki.wooledge.org/BashFAQ/031#np2">(**)</a></td><td>[[ -n $var &#x26;&#x26; -f $var ]] &#x26;&#x26; echo "$var is a file"</td></tr><tr><td></td><td>||</td><td>-o <a href="https://mywiki.wooledge.org/BashFAQ/031#np2">(**)</a></td><td>[[ -b $var </td></tr><tr><td>expression grouping</td><td>(...)</td><td>\( ... \) <a href="https://mywiki.wooledge.org/BashFAQ/031#np2">(**)</a></td><td>[[ $var = img* &#x26;&#x26; ($var = *.png </td></tr><tr><td>Pattern matching</td><td>= (or ==)</td><td>(not available)</td><td>[[ $name = a* ]] </td></tr><tr><td><a href="https://mywiki.wooledge.org/RegularExpression">RegularExpression</a> matching</td><td>=~</td><td>(not available)</td><td>[[ $(date) =~ ^Fri\ ...\ 13 ]] &#x26;&#x26; echo "It's Friday the 13th!"</td></tr></tbody></table>
