# packages

- [vim packages](https://vimhelp.org/repeat.txt.html#packages)

vim now has packages integrated since version 8
folder is `~/.vim/pack/`
packages need to have a start subfolder in that dir so
`~/.vim/pack/*/start/`

## git tool fugitive

- [vim-fugitive](https://github.com/tpope/vim-fugitive)

### install

```bash
mkdir -p ~/.vim/pack/tpope/start
cd ~/.vim/pack/tpope/start
git clone https://tpope.io/vim/fugitive.git
vim -u NONE -c "helptags fugitive/doc" -c q
```

### usage

```bash
vim -c ":help fugitive"
```

```vim
:help :Gwrite
:help :Git
:Git status
:Git add python/ vim/
:Git commit
:Git push
```

| git               | fugitive   | action                                                   |
| ----------------- | ---------- | -------------------------------------------------------- |
| `:Git add %`      | `:Gwrite`  | Stage the current file to the index                      |
| `:Git checkout %` | `:Gread`   | Revert current file to last checked in version           |
| `:Git rm %`       | `:Gremove` | Delete the current file and the corresponding Vim buffer |
| `:Git mv %`       | `:Gmove`   | Rename the current file and the corresponding Vim buffer |


