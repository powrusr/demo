# line selection 
## first last
```bash
sed -n '1p' /etc/passwd
root:x:0:0:root:/root:/bin/bash

## last line
sed -n '$p' /etc/passwd
_flatpak:x:130:139:Flatpak system-wide installation helper,,,:/nonexistent:/usr/sbin/nologin
```

## range

```bash
sed -n '1,3p' /etc/passwd
root:x:0:0:root:/root:/bin/bash
daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin
bin:x:2:2:bin:/bin:/usr/sbin/nologin

sed -n '48,$p' /etc/passwd
duke:x:1000:1000:duke,,,:/home/duke:/bin/bash
lxc-dnsmasq:x:129:137:LXC dnsmasq,,,:/var/lib/lxc:/usr/sbin/nologin
_flatpak:x:130:139:Flatpak system-wide installation helper,,,:/nonexistent:/usr/sbin/nologin

# last line minus one hack
sed -n "45,$(( $(wc -l </etc/ssh/ssh_config)-1 ))p" < /etc/ssh/ssh_config
```

### fetch blocks

```bash
sed -n '/<Directory/,/<\/Directory/p' /etc/apache2/conf-available/javascript-common.conf 
<Directory "/usr/share/javascript/">
    Options FollowSymLinks MultiViews
</Directory>
```

# append insert delete actions

## append
```bash
# doesn't append yet since no -i inplace flag
sudo sed '$a 208.67.222.222 opendns' /etc/hosts
...
ff00::0 ip6-mcastprefix
ff02::1 ip6-allnodes
ff02::2 ip6-allrouters
208.67.222.222 opendns

# append line on each line in range
sed "45,$(( $(wc -l </etc/ssh/ssh_config)-1 ))a # comment above each line in range" < /etc/ssh/ssh_config
#   EscapeChar ~
#   Tunnel no
#   TunnelDevice any:any
# comment above each line in range
#   PermitLocalCommand no
# comment above each line in range
#   VisualHostKey no
# comment above each line in range
#   ProxyCommand ssh -q -W %h:%p gateway.example.com
# comment above each line in range
#   RekeyLimit 1G 1h
# comment above each line in range
#   UserKnownHostsFile ~/.ssh/known_hosts.d/%k
# comment above each line in range
    SendEnv LANG LC_*
# comment above each line in range
    HashKnownHosts yes
# comment above each line in range
    GSSAPIAuthentication yes

# append to last line
sed "$ a # last in line" < /etc/ssh/ssh_config

    HashKnownHosts yes
    GSSAPIAuthentication yes
# last in line

```
### with regex

sed '/PATTERN/ a <LINE-TO-BE-ADDED>' FILE.txt

```bash
$ sed '/5/ a #Next line is the 6th line, not this' sedtest.txt
This is line #1
This is line #2
This is line #3
This is line #4
This is line #5
#Next line is the 6th line, not this
This is line #6
```

## insert

```bash
sed '/8/ i #This line is inserted using sed' sedtest.txt
This is line #1
This is line #2
This is line #3
This is line #5
This is line #6
This is line #7
#This line is inserted using sed
This is line #8
This is line #9
This is line #10
```

### insert script

```bash
#!/bin/bash
for f in  ~/scripts/*.sh; do
  firstline=$(sed -n '1p' $f)
  if [[ $firstline != "#!"* ]]; then
    echo "adding shebang to $f"
    sed -i '1i #!/bin/bash' $f
  fi
done
```

# substitute

## ignore case

```bash
sed -E -i 's/something/else/Ig' file.py

# test with grep -i "something" file.py
```

## backreferences

```bash
echo 'one two three' | sed 's/\(one\) \(two\) \(three\)/\3 \2 \1/'
three two one
```

When using extended regular expressions parenthesis do grouping by default, and dont have to be escaped

```bash
echo 'Hello world!' | sed -E 's/(Hello) world!/\1 sed/'
Hello sed
```

```bash
echo Hello 123 reg_exp | sed -E 's/(\w+) (\w+) (\w+)/\3 \2 \1/'
reg_exp 123 Hello
echo Hello 123 reg_exp | sed -E 's/([[:alnum:]]+) ([[:alnum:]]+) ([[:alnum:]]+)/\3 \2 \1/'
reg 123 Hello_exp # underscore wasnt matched
echo Hello 123 reg_exp | sed -E 's/([[:alnum:]]+) ([[:alnum:]]+) ([[:alnum:]_]+)/\3 \2 \1/'
reg_exp 123 Hello

# The sequence \w is equivalent to [[:alnum:]_]
```
# grouping expressions

## by semicolon

```bash
# del empty line, del commented line
sed '/^$/d;/^#/d'
```

## file scripts

### simple

```bash
vim simple.sed
/^$/d
/^#/d

sed -f simple.sed ssh_config
```
### advanced

#### branching & labels

keep clients alive

ClientAliveInterval: how often is a keep alive packet sent to client
ClientAliveCountMax: intervals that can pass without a client response

ssh_client.sed:
t = test

```bash
#!/bin/sed -Ef
/ClientAliveInterval/ {                 # string to be searched
  s/^(ClientAliveInterval).*$/\1 90/
  t countMax                               # if substitution success goto label countMax
  s/.*/ClientAliveInterval 90/          # eg started w comment = full replacement
  t countMax
}

:countMax
/ClientAliveCountMax/ {
  s/^(ClientAliveCountMax).*$/\1 3/
  t del
  s/.*/ClientAliveCountMax 3/
  t del
}

:del
/^($|#)/del                             # delete empty lines
```
#### in-place backups

```bash
sed -f ssh_client.sed -i.bak ssh_config

```

#### using sed in a loop
ssh -n: redirect stdin to /dev/null, allowing remote execution on multiple servers

```bash
while read server; do
    echo 'configuring server: $server'
    ssh -n -C $server "sudo sed -Ei.$(date +%F) '/^($|#)/d' /etc/ssh/sshd_config"
    echo 'done configuring $server'
done < server_ips.txt
```
