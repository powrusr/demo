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
