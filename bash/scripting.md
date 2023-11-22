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

## math

### Arithmetic Expansion

POSIX sh (and all shells based on it, including Bash and ksh) uses the $(( )) syntax to do arithmetic, using the same syntax as C. (See the Bash hackers article for the full syntax.) Bash calls this an "Arithmetic Expansion", and it obeys the same basic rules as all other $... substitutions.

$(( )) is the first example of a math context, meaning a context where the syntax and semantics of C's integer arithmetic are used. This will be discussed in more detail below.

Here are a few examples using $(( )):


```bash
i=$((j + 3))
lvcreate -L "$((24 * 1024))" -n lv99 vg99
q=$((29 / 6)) r=$((29 % 6))
if test "$((a%4))" = 0; then ...
```

### Arithmetic Commands

Bash also offers two forms of commands that use a math context. The expressions within the math context follow the same rules as the expressions inside $(( )), but since we have a command, we also get an exit status, and side effects.

The exit status is based on the value of the (last) expression. If the expression evaluates to 0, the command is considered "false" and returns 1. Otherwise, the command is considered "true" and returns 0. See below for examples of this.

The first arithmetic command is let:

```bash
let a=17+23
echo "a = $a"      # Prints a = 40
```

Note that each arithmetic expression has to be passed as a single argument to the let command, so you need quotes if there are spaces or globbing characters. Thus:

```bash
let a=17 + 23      # WRONG
let a="17 + 23"    # Right
let 'a = 17 + 23'  # Right
let a=17 a+=23     # Right (2 arithmetic expressions)

let a[1]=1+1       # Wrong (try after touch a1=1+1 or with shopt -s failglob)
let 'a[1]=1+1'     # Right
let a\[1\]=1+1     # Right, but very odd.
```

The second command is (( )). It works identically to let, but you don't need to quote the expressions because they are delimited by the (( and )) syntax. For example:


```bash
((a=$a+7))         # Add 7 to a
((a = a + 7))      # Add 7 to a.  Identical to the previous command.
((a += 7))         # Add 7 to a.  Identical to the previous command.

((a = RANDOM % 10 + 1))     # Choose a random number from 1 to 10.
                            # % is modulus, as in C.

if ((a > 5)); then echo "a is more than 5"; fi
```

(( )) is used more widely than let, because it fits so well into an if or while command.

> When Bash runs a command, that command will return an exit status from 0 to 255. 0 is considered "success" (which is "true" when used in the context of an if or while command). However, when evaluating an arithmetic expression, C language rules (0 is false, anything else is true) apply.

Some examples:

```bash 
true; echo "$?"       # Writes 0, because a successful command returns 0.
((10 > 6)); echo "$?" # Also 0.  An arithmetic command returns 0 for true.
echo "$((10 > 6))"    # Writes 1.  An arithmetic expression evaluates to 1 for true.
```

In addition to a comparison returning 1 for true, an arithmetic command that evaluates to any non-zero value returns true as an exit status.

```bash
if ((1)); then echo true; fi     # Writes true.
```

This also lets you use "flag" variables, just like in a C program:

```bash
found=0
while ...; do
  ...
  if something; then found=1; fi    # Found one!  Keep going.
  ...
done
if ((found)); then ...
```

This also means that every arithmetic command we run is returning an exit status. Most scripts ignore these, but if you are using set -e you may be unpleasantly surprised when your program aborts because you assigned 0 to a number.

### Math Contexts

As we've seen, the insides of $(( )) and (( )), and the arguments of let, are math contexts.

Also, (non-associative) array indices are a math context:


```bash
n=0
while read line; do
   array[n++]=$line      # array[] forces a numeric context
done
```
In case it wasn't obvious, the (( )) in a C-style for command are a math context. Or three separate math contexts, depending on your point of view.


```bash
for ((i=0, j=0; i<100; i++, j+=5)); do ...
```
Finally, the start and length arguments of Bash's ${var:start:length} parameter expansion are math contexts.


```bash
echo "${string:i+2:len-2}"
```
