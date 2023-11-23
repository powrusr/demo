# packages

https://vimhelp.org/repeat.txt.html#packages

vim now has packages integrated since version 8
folder is `~/.vim/pack/`
packages need to have a start subfolder in that dir so
`~/.vim/pack/*/start/`

## git tool fugitive

### install

```bash
mkdir -p ~/.vim/pack/tpope/start
cd ~/.vim/pack/tpope/start
git clone https://tpope.io/vim/fugitive.git
vim -u NONE -c "helptags fugitive/doc" -c q
```

### usage

```vim
:Git status
:Git add
```
