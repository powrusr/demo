# BASH

## find

### regex & delete

[posix extended regex](https://www.gnu.org/software/findutils/manual/html_node/find_html/posix_002dextended-regular-expression-syntax.html)

```bash
# check no of affected files first
find . -regex '.*(1).mp4' | wc -l

# also catch (2).mp4 etc
find . -regex '.*([0-9]+).mp4' | wc -l

# also include subtitle files
find . -regex '.*([0-9]+).mp4\|.*en.srt' | wc -l
find . -regextype sed -regex '.*([[:digit:]]\+).mp4\|.*en.srt' | wc -l

# delete with -delete flag, no need for -exec rm '{}' \;
find . -regextype sed -regex '.*([[:digit:]]\+).mp4\|.*en.srt' -delete
```


