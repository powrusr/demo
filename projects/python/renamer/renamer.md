# recursively clean & rename filenames in a folder

[code](https://raw.githubusercontent.com/powrusr/docs/main/projects/python/renamer/recursively_rename.py)

```python
from pathlib import Path
import re  # https://docs.python.org/3/howto/regex.html
import pprint as pp

# https://realpython.com/get-all-files-in-directory-python/#recursively-listing-with-rglob
# https://docs.python.org/3/library/pathlib.html
# https://jex.im/regulex/#!flags=&re=%5Babc%5D
home = Path.home()
videos_path = home / "Videos"


def clr_domains(line: str, replace_with: str = ''):
    # p_domain = r"\w+\.(com|org|me)\s*\b"
    p_domain = r"[\w_]+\.(com|org|me)[_\s*\w]?"
    return re.compile(p_domain, flags=re.IGNORECASE).sub(repl=replace_with, string=line)


def clr_words(line: str, replace_with: str = ''):
    words = ["udemy", "galaxy", "pmedia", "linkedin", "webrip", "x26", "DD5.1", "elite", "720", "1080p"]
    # matches wordabc[a12] or wordx(123)
    # pattern = r'linkedin[a-zA-Z0-9()[\]]*'
    return re.compile(r'%s' % r'[a-zA-Z0-9()[\]]*|'.join(map(re.escape, words)),
                      flags=re.IGNORECASE).sub(replace_with, line)


def clr_leading(line: str, replace_with: str = ''):
    p_leading = r"(?<=^)[\s&~_|\\\/+-]+"
    return re.compile(p_leading).sub(repl=replace_with, string=line)


def clr_trailing(line: str, replace_with: str = ''):
    p_trailing = r"[\s_|\\\/&+-.]+$"
    return re.compile(p_trailing).sub(repl=replace_with, string=line)


def clr_unwanted(line: str, replace_with: str = ''):
    p_unwanted_chars = r"[｜&|+[\]:;\\,()'\"]"  # don't add _
    return re.compile(p_unwanted_chars).sub(repl=replace_with, string=line)


def clr_youtube_id(line: str, replace_with: str = ''):
    p_youtube_id = r"(?=\s+\[?\w{11}(]*))(\s+\[?\w+(([A-Z]+\d+)|(\d+[a-zA-Z]))\w+\]*)"  # also matches id in brackets
    return re.compile(p_youtube_id).sub(repl=replace_with, string=line)


def clr_dumb_dot(line: str, replace_with: str = '_'):
    p_dumb_dot = r"(?<=\d)(\.\s+)"
    return re.compile(p_dumb_dot).sub(repl=replace_with, string=line)


def clr_lone_chars(line: str, replace_with: str = '_'):
    p_lone_character = r"\b(\s[^a ]\s)\b|(\s[-]\s)"
    return re.compile(p_lone_character).sub(repl=replace_with, string=line)


def clr_spaces(line: str, replace_with: str = '_'):
    p_spaces = r"( +)"
    return re.compile(p_spaces).sub(repl=replace_with, string=line)


def clr_filesize(line: str, replace_with: str = ''):
    p_filesize = r"(\d{3,}mb)"
    return re.compile(p_filesize, re.IGNORECASE).sub(repl=replace_with, string=line)


def trim_length(line: str, limit: int = 75):
    return line[:limit]


def rm_duplicates(paths: [Path]):
    # file that should already exist is group 1 part
    duplicates = []
    p_duplicate = re.compile(r"(.+)\(([0-9]|copy)\)$")
    duplicates = [file for file in paths if re.match(p_duplicate, str(file.stem))]
    deleted = []
    for duplicate in duplicates:
        original_stem = re.sub(pattern=p_duplicate, repl=r"\1", string=duplicate.stem)
        original_path = duplicate.parent / (original_stem + duplicate.suffix)
        # check rebuilt original file without (#) or (copy) part really exists, only then delete
        if original_path.is_file():
            # if missing_ok=True, FileNotFoundError exceptions will be ignored
            Path.unlink(duplicate)
            deleted.append(duplicate)
    return deleted


# todo: rename (1) files in LFCS folder


def clean_stem(line: str):
    return (  # don't mess with the order
        clr_trailing(
            trim_length(
                clr_spaces(
                    clr_dumb_dot(
                        clr_leading(  # again
                            clr_filesize(
                                clr_domains(
                                    clr_words(
                                        clr_youtube_id(
                                            clr_unwanted(
                                                clr_lone_chars(
                                                    clr_leading(line)
                                                ))))))))))))


def stem_of_dir(line: str):
    # text between last / and EOL or 2nd to last / and final /
    p_stem = r"(?:.+\/)(.+?)(?:\/|$)"
    return re.compile(p_stem).match(line).group(1)


def clean_files(paths: [Path]):
    cleaned_path = None
    files = [f for f in paths if Path.is_file(f)]
    # files include directory so need to rename them first
    for f in files:
        stem = f.stem
        cleaned_stem = clean_stem(stem)
        cleaned_path = f.parent / (cleaned_stem + f.suffix)
        f.rename(target=cleaned_path)
    # rebuild paths as file paths have now been renamed
    renewed_paths = get_folders(videos_path)
    all_dirs = [d for d in renewed_paths if Path.is_dir(d)]
    # get longest dir

    def dir_parts_length_generator(dirs: [Path]):
        for d in dirs:
            length = len(d.parts)
            yield length

    dirs_part_lengths = list(dir_parts_length_generator(all_dirs))
    idx_of_deepest_dir = dirs_part_lengths.index(max(dirs_part_lengths, key=int))
    deepest_dir = all_dirs[idx_of_deepest_dir]
    # set parent depth from which to start renaming directories
    # start from highest depth end with lowest depth to avoid FileNotFoundErrors due to renaming
    max_depth = len(deepest_dir.parts)
    min_depth = len(videos_path.parts)
    current_level = max_depth  # start in deepest folder depth
    while current_level > min_depth:  # stop when we reach folder we start from
        current_level_dirs = [d for d in all_dirs if len(d.parts) == current_level]
        for d in current_level_dirs:
            # current_path = Path(str(deepest_dir.parts[0:current_level]))
            stem = stem_of_dir(str(d))  # reliably process ext with regex in dirs
            cleaned_stem = clean_stem(stem)
            cleaned_path = d.parent / cleaned_stem
            d.rename(target=cleaned_path)
        current_level -= 1


def get_folders(path: Path) -> [Path]:
    return list(path.rglob("./"))


def get_files(path: Path) -> [Path]:
    return list(path.rglob("*"))


if __name__ == '__main__':
    print("following duplicates were removed:")
    pp.pprint(rm_duplicates(get_files(videos_path)))
    # clean files
    clean_files(get_files(videos_path))

"""
# another way to get different depth of folders using glob
 def get_subitems(folder: Path, level: int) -> List[Path]:
     if level == 0:
         return [folder]
     pattern = "/".join(["*"] * level)
    return sorted(folder.glob(pattern))
"""
```

[tests](https://raw.githubusercontent.com/powrusr/docs/main/projects/python/renamer/tests/test_recursively_rename.py)

```python
import pytest
import recursively_rename
import tempfile
import pathlib

@pytest.mark.parametrize("given, expected", [
    ("[ Course.com ] Learning - Learning Radio", "[ ] Learning - Learning Radio"),
    ("[[CoursesOnline.OrG]] Terraform - Getting Started", "[[]] Terraform - Getting Started"),
    ("[ CourseWikia.com ] Linkedin - Learning Rsync", "[ ] Linkedin - Learning Rsync")
])
def test_clr_domains(given: str, expected: str):
    assert recursively_rename.clr_domains(line=given) == expected


@pytest.mark.parametrize("given, expected, replace_char", [
    ("&    yet more ", "yet more ", ""),
    ("~ tiltme ", "tiltme ", ""),
    ("--  ending", "ending", ""),
    ("+ - | word", "word", ""),
    ("/ word", "word", ""),
    ("___word", "word", ""),
])
def test_clr_leading(given: str, expected: str, replace_char: str):
    assert recursively_rename.clr_leading(line=given, replace_with=replace_char) == expected


@pytest.mark.parametrize("given, expected, replace_char", [
    ("trailing space ", "trailing space", ""),
    ("trailing underspace_ ", "trailing underspace", ""),
    ("trailing triple_under___", "trailing triple_under", ""),
    ("trailing amper &", "trailing amper", ""),
    ("ok/man_", "ok/man", ""),
    ("Butchers.Crossing.2023.720p...", "Butchers.Crossing.2023.720p", "")
])
def test_clr_trailing(given: str, expected: str, replace_char: str):
    assert recursively_rename.clr_trailing(line=given, replace_with=replace_char) == expected


@pytest.mark.parametrize("given, expected, replace_char", [
    ("strip some&", "strip some", ""),
    ("unw+nted", "unwOnted", "O"),
    ("ok\man_", "okman_", ""),
    ("no;no:no\\no[", "nononono", ""),
    ("no;(no):no\\no[", "nononono", ""),
])
def test_clr_unwanted(given: str, expected: str, replace_char: str):
    assert recursively_rename.clr_unwanted(line=given, replace_with=replace_char) == expected


@pytest.mark.parametrize("given, expected, replace_char", [
    ("Andy Constan CR8zIcRyq9A", "Andy Constan", ""),
    ("LED Strip Lights Ay4G6RasAek", "LED Strip Lights", ""),
    ("Arthur Hayes RVgdD3Av47A", "Arthur Hayes", ""),
    ("Landing Operation goes well [MmSV3YQAY8g]", "Landing Operation goes well", ""),
])
def test_clr_youtube_id(given, replace_char, expected):
    assert recursively_rename.clr_youtube_id(line=given, replace_with=replace_char) == expected


@pytest.mark.parametrize("given, expected, replace_char", [
    ("lone & amper", "lone_amper", "_"),
    ("lone ~ tilde", "lonetilde", ""),
    ("lone | pipe", "lone pipe", " "),
    ("lone * star", "lone_star", "_"),
    ("once upon a time", "once upon a time", ""),
    ("once_upon b time", "once_upon_time", "_"),
    ("2 - (A)synchronous Python", "2_(A)synchronous Python", "_")
])
def test_clr_lone_chars(given, replace_char, expected):
    assert recursively_rename.clr_lone_chars(line=given, replace_with=replace_char) == expected


@pytest.mark.parametrize("given, expected, replace_char", [
    (" odd spaces ", "_odd_spaces_", "_"),
    ("deep       _        space", "deep_space", ""),
])
def test_clr_spaces(given, replace_char, expected):
    assert recursively_rename.clr_spaces(line=given, replace_with=replace_char) == expected


@pytest.mark.parametrize("given, expected", [
    ("004. 13.3 Modifying Service Configuration", "004_13.3 Modifying Service Configuration"),
    ("006. 3.5 Managing 5. Systemd Unit Dependencies en", "006_3.5 Managing 5_Systemd Unit Dependencies en"),
])
def test_clr_dumb_dot(given, expected):
    assert recursively_rename.clr_dumb_dot(line=given) == expected


@pytest.mark.parametrize("given, expected", [
    ("/home/duke/Videos/Oppenheimer.2023", "Oppenheimer.2023"),
    ("/home/duke/Videos/Oppenheimer.2023/", "Oppenheimer.2023"),
    ("/home/duke/Videos/Python Logging Step by Step Intro/1. Step by step introduction to Python Logging module/",
     "1. Step by step introduction to Python Logging module")
])
def test_stem_of_dir(given: str, expected: str):
    assert recursively_rename.stem_of_dir(line=given) == expected


@pytest.mark.parametrize("given, expected", [
    ("Udemy Using AI to troubleshoot Linux", " Using AI to troubleshoot Linux"),
    ("Linkedin - Learning Rsync", " - Learning Rsync"),
    ("WEBRip.1400MB.DD5.1.x264-GalaxyRG", ".1400MB..-"),
    ("WEBRip.1400MB.DD5.1.720p", ".1400MB.."),
])
def test_clr_words(given: str, expected: str):
    assert recursively_rename.clr_words(line=given) == expected


@pytest.mark.parametrize("given, expected", [
    ("WEBRip.1400MB", "WEBRip."),
])
def test_clr_filesize(given: str, expected: str):
    assert recursively_rename.clr_filesize(line=given) == expected


@pytest.mark.parametrize("given, expected", [
    ("test _ some + 2022 text | ", "test_some_2022_text"),
    ("Financial Conditions Are Easing Janet Yellen's Fueling A Massive Stock Market Rally ｜ Andy Constan CR8zIcRyq9A",
     "Financial_Conditions_Are_Easing_Janet_Yellens_Fueling_Massive_Stock_Market"),
    ("006. 3.5 Managing 5. Systemd Unit Dependencies en", "006_3.5_Managing_5_Systemd_Unit_Dependencies_en"),
    ("very super long length line to really like over do it, that its become ridiculous haha seriously bruh",
     "very_super_long_length_line_to_really_like_over_do_it_that_its_become_ridic"),
    ("[[CoursesOnline.OrG]] Terraform - Getting Started", "Terraform_Getting_Started"),
    ("[ CourseWikia.com ] Linkedin - Learning Rsync", "Learning_Rsync"),
    ("2 - (A)synchronous Python", "2_Asynchronous_Python"),
    ("Butchers.Crossing.2023.720p...", "Butchers.Crossing.2023")
])
def test_clean_all(given: str, expected: str):
    assert recursively_rename.clean_stem(line=given) == expected
```

