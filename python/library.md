# code library

## adding files to a list & compression

```python
from pathlib import Path
import contextlib
import os


def file_exists(file_path):
    return Path(file_path).exists()


def compress_to_bz2_file(source_file, output_file, compression_lvl=9):
    import bz2
    if 1 <= compression_lvl <= 9:
        with open(source_file, 'rb') as data:
            tar_bz2_contents = bz2.compress(data.read(), compresslevel=compression_lvl)
    else:
        raise ValueError("compression level has to be between 1 and 9")


# concatenate_files_in_list(text_files_list, final_file, 'latin-1', True)
def concatenate_files_in_list(file_list, output_file, encode_in='utf-8', is_small=False, use_shutil=False):
    # don't forget to specify target folder when opening files!
    # todo: check if file was created already
    encoding_format = encode_in
    print(f"target file -> {output_file}")
    readmode = ('rb' if use_shutil else 'r')
    writemode = ('wb' if use_shutil else 'w+')
    with open(output_file, writemode) as finaltext:
        for file_name in file_list:
            print(f"currently appending file: {file_name}")
            with open(file_name, readmode, encoding=encoding_format) as file_currently_opened:
                if use_shutil:
                    import shutil
                    shutil.copyfileobj(file_name, finaltext)
                elif is_small:
                    finaltext.write(file_currently_opened.read())
                elif not is_small:
                    for line in file_currently_opened:
                        finaltext.write(line)


def add_files_from_folder_to_list(filetype, folder_target='./'):
    folder = set_folder(folder_target)
    print(folder)
    print(f"getting files from {folder} of type {filetype} ...")
    all_files = []
    for path, dirs, files in os.walk(folder):
        for filename in files:
            if filename.endswith('.txt'):
                full_path = os.path.join(path,filename)
                all_files.append(full_path)
    with contextlib.suppress(Exception):
        raise RuntimeError('something went wrong')
    return all_files


def set_folder(folder_dir):
    isdir = Path(folder_dir)
    if not isdir:
        raise Exception(f"folder you're trying to set is invalid -> {folder_dir}")
    absolute_path = isdir.resolve()
    return absolute_path


def test_files_in_list(file_list: []):
    print("testing files in list... ")
    for file in file_list:
        if not Path(file).exists():
            print(f"oops: {file} doesn't seem to exist")
```
