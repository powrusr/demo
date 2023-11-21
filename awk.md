# basics
[html manual](https://www.gnu.org/software/gawk/manual/html_node/)

## begin-main-end block

### main

- executed for each input record

```bash
# main block with search regex & printing record number and complete line ($0)
awk -F: '/bin\/bash$/{ printf "%2d%s%s\n",NR,": ",$0 }' /etc/passwd

 1: root:x:0:0:root:/root:/bin/bash
48: duke:x:1000:1000:duke,,,:/home/duke:/bin/bash
# 
```

### begin

- print headers
- set FS variable
- declare your own vars

```bash
awk 'BEGIN { FS=":" } /bin\/bash$/{ printf "%2d%s%5s%5d\n",NR,":",$1,$3}' /etc/passwd

 1: root    0
48: duke 1000
```
note the extra %s to account for the ":" 

```bash
awk 'BEGIN { FS=":";COUNT=0 } /bin\/bash$/{ COUNT++; printf "%8s%2d%s%2d%s%5s%5d\n","row no: ",NR," count:",COUNT,":",$1,$3}' /etc/passwd

row no:  1 count: 1: root    0
row no: 48 count: 2: duke 1000

```

### end

```bash
awk 'BEGIN {FS=":"; printf "%3s%12s%6s\n","No: ","Username","UID";COUNT=0} /bin\/bash/{COUNT++; printf "%4s%s%10s%6d\n",COUNT,": ",$1,$3;} END {print "there are " NR " users, of whom " COUNT " use bash"}' /etc/passwd

No:     Username   UID
   1:       root     0
   2:       duke  1000
there are 50 users, of whom 2 use bash

```
## special variables

[controlling vars](https://www.gnu.org/software/gawk/manual/html_node/User_002dmodified.html)
[other vars](https://www.gnu.org/software/gawk/manual/html_node/Auto_002dset.html)
[Using ARGC and ARGV](https://www.gnu.org/software/gawk/manual/html_node/ARGC-and-ARGV.html)

- NR: number of input records awk has processed since the beginning of the program’s execution
- NF: The number of fields in the current input record. NF is set each time a new record is read, when a new field is created, or when $0 changes
- FS: field seperator, defaults to a space " " (is set with -F flag)
- RS: record seperator
- OFS: seperator at output time
- ORS: output at end of every print statement, \n by default
### custom FS OFS

```bash
awk 'BEGIN {FS=":"; OFS=","} { print $1,$2,$3,$4,$6,$7 }' /etc/passwd | sed -n '1,5p'
"""
root,x,0,0,/root,/bin/bash
daemon,x,1,1,/usr/sbin,/usr/sbin/nologin
bin,x,2,2,/bin,/usr/sbin/nologin
sys,x,3,3,/dev,/usr/sbin/nologin
sync,x,4,65534,/bin,/bin/sync
"""
```
### custom record seperator

```bash
#!/bin/awk -f
BEGIN { RS="\\n\\s*\\n" }
$0 ~ pattern { print }
```

## printf
printf allows for precise control of output

[printf control letters](https://www.gnu.org/software/gawk/manual/html_node/Control-Letters.html)
[printf examples](https://www.gnu.org/software/gawk/manual/html_node/Printf-Examples.html)

## sprintf

sprintf.awk

```bash
BEGIN {
  for (i = 97; i <= 122; i++) {
    char = sprintf( "%c", i );
    printf("%s ", char )
  }
  print
}
```

```bash
awk -f sprintf.awk 
a b c d e f g h i j k l m n o p q r s t u v w x y z 
```

## begin main end block
```bash

```

## control structures

note sprintf is same as printf but returns string that can be assigned to a variable

```bash
BEGIN {
    for (i = 1; i < ARGC; i++) {
        if (ARGV[i] == "-v")
            verbose = 1
        else if (ARGV[i] == "-q")
            debug = 1
        else if (ARGV[i] ~ /^-./) {
            e = sprintf("%s: unrecognized option -- %c",
                    ARGV[0], substr(ARGV[i], 2, 1))
            print e > "/dev/stderr"
        } else
            break
        delete ARGV[i]
    }
}
```

# scripts

## bash users report

```bash
#!/bin/awk -f
BEGIN {
  FS=":";
  printf  "%4s%20s%6s\n","Num:", "username", "UID";
  COUNT=0
}

/bash$/{
  COUNT++;
  printf "%2d%s%20s%6d\n",COUNT, ": ", $1, $3;
}

END {
   print "there are " NR " users, and " COUNT " use bash";
}
```

## report users that logged in

```bash
!(/Never logged in/ || /^Username/ || /^root/) {
  COUNT++;
  if ( NF == 8 )
    printf "%8s %2s %3s %4s\n", $1,$5,$4,$8;
  else
    printf "%8s %2s %3s %4s\n", $1,$6,$5,$9};
END {
  print "======================";
  print "No of usernames processed: ", COUNT;
}
```

## extract ip from failed ssh login

```bash
#!/bin/awk -f
/Failed password for invalid user /{
        ips[$13]=1   # stored on index, 1 is meaningless
}
/Failed password for root /{
        ips[$11]=1
}
END {
        n=asorti(ips, sorted_ips)
        for(i=1; i<=n; i++) {
                print sorted_ips[i]
        }
}
```

## convert passwd to yaml

```bash
BEGIN {
  FS=":";
  print "users: "
}
$0 ~ pattern {
  for (i=1; i<=NF; i++) {
    switch(i) {
        case 1:
            printf "  - user: %s\n",$i;
            break;
        case 2:
            printf "    password: %s\n",$i;
            break;
        case 3:
            printf "    uid: %s\n",$i;
            break;
        case 4:
            printf "    gid: %s\n",$i;
            break;
        case 5:
            printf "    gecos: %s\n",$i;
            break;
        case 6:
            printf "    home: %s\n",$i;
            break;
        case 7:
            printf "    shell: %s\n",$i;
            break;
    }
  }
}
```
