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
