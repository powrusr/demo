# ansible

## setup

### create dedicated python environment

helpful package to work with python environments
[virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/)

```bash
sudo apt-get install virtualenvwrapper virtualenvwrapper-doc
```

verify virtualenvwrapper.sh location

```bash
sudo find /usr -type f -name virtualenvwrapper.sh
/usr/share/virtualenvwrapper/virtualenvwrapper.sh
```

add to bashrc

```bash
export WORKON_HOME=$HOME/.virtualenvs
export PROJECT_HOME=$HOME/Devel
source /usr/share/virtualenvwrapper/virtualenvwrapper.sh
```

reload startup file to create scripts

```bash
$ . ~/.bashrc 
virtualenvwrapper.user_scripts creating /home/powrusr/.virtualenvs/premkproject
virtualenvwrapper.user_scripts creating /home/powrusr/.virtualenvs/postmkproject
virtualenvwrapper.user_scripts creating /home/powrusr/.virtualenvs/initialize
virtualenvwrapper.user_scripts creating /home/powrusr/.virtualenvs/premkvirtualenv
virtualenvwrapper.user_scripts creating /home/powrusr/.virtualenvs/postmkvirtualenv
virtualenvwrapper.user_scripts creating /home/powrusr/.virtualenvs/prermvirtualenv
virtualenvwrapper.user_scripts creating /home/powrusr/.virtualenvs/postrmvirtualenv
virtualenvwrapper.user_scripts creating /home/powrusr/.virtualenvs/predeactivate
virtualenvwrapper.user_scripts creating /home/powrusr/.virtualenvs/postdeactivate
virtualenvwrapper.user_scripts creating /home/powrusr/.virtualenvs/preactivate
virtualenvwrapper.user_scripts creating /home/powrusr/.virtualenvs/postactivate
virtualenvwrapper.user_scripts creating /home/powrusr/.virtualenvs/get_env_details
```

create your environment and activate it

```bash
# create environment
mkvirtualenv project

# move into environment
workon project

# exit environment
deactivate
```
### install ansible

install [ansible](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html)

```bash
pip3 install ansible argcomplete

# validate your ansible installation:
which ansible
/home/powrusr/.virtualenvs/test/bin/ansible

ansible --version
ansible [core 2.16.0]
ansible-community --version
Ansible community version 9.0.1

# enable argcomplete
activate-global-python-argcomplete
. ~/.bash_completion

ansible --
--args                      --diff                      --scp-extra-args
--ask-become-pass           --extra-vars                --sftp-extra-args
--ask-pass                  --forks                     --ssh-common-args
--ask-vault-pass            --help                      --ssh-extra-args
--ask-vault-password        --inventory                 --task-timeout
--background                --inventory-file            --timeout
--become                    --key-file                  --tree
--become-method             --limit                     --user
--become-pass-file          --list-hosts                --vault-id
--become-password-file      --module-name               --vault-pass-file
--become-user               --module-path               --vault-password-file
--check                     --one-line                  --verbose
--connection                --playbook-dir              --version
--connection-password-file  --poll                      
--conn-pass-file            --private-key               
```


### generate config

```bash
ansible-config init --disabled -t all > ansible.cfg
# or use default ~/$HOME/.ansible.cfg
```

### generate vault password

[generate password](https://docs.ansible.com/ansible/latest/reference_appendices/faq.html#how-do-i-generate-encrypted-passwords-for-the-user-module)

```bash
$ pip3 install passlib
Collecting passlib
  Downloading passlib-1.7.4-py2.py3-none-any.whl (525 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 525.6/525.6 KB 5.3 MB/s eta 0:00:00
Installing collected packages: passlib
Successfully installed passlib-1.7.4
(test) duke@nukem:~$ python -c "from passlib.hash import sha512_crypt; import getpass; print(sha512_crypt.using(rounds=5000).hash(getpass.getpass()))"
Password: 
$6$48pDj8lg7.79c1rl$pqzCshW2wjBYhc77tcPdyWfPMVYFOiL0rgcGYnfNq3Gy6dFrOTNB2f4DAO8rpRDVrlv2vYrRt7ICPEda7RT3G1
# not actually used, just for demo
```

### configure vault variables

```yaml
   ansible_become_pass: yOurp455w0rD
   admin: serveradmin
   admin_password: 'hash of serveradmin password'
   local_admin: localhost_admin  # admin you're running playbook from
   ssh:
     port: 12345
     private: "~/.ssh/myprivatekey"
     public: "ssh-rsa AAAAmypublickeycontenthere"
```

note: create group_vars/hostgroup.yml files with ansible-vault should they contain sensitive data

### tweak config

adjust defaults in `ansible.cfg` as needed

```ini
[defaults]
inventory=~/.ansible/hosts
private_key_file=~/.ssh/your_private_key
ask_vault_pass=True
```

### tweak templates in roles to personal preferences

```bash
roles/user_profiles/templates/.vimrc.j2
roles/user_profiles/templates/.bash_aliases.j2
roles/user_profiles/templates/.bash_functions.j2
```

setting `ask_vault_pass=True` tells ansible to ask you for the password whenever a playbook loads your encrypted variables


## create encrypted file for sensitive data

```bash
ansible-vault create vault/secrets.yml
```

## project folder structure

example of a projects folder structure

```bash
$ tree cere/
cere/
├── ansible.cfg
├── cere_validator.yml
├── files
│   ├── rpc_methods.json
│   └── sshd_config
├── get_server_info.yml
├── group_vars
│   ├── all.yml
│   ├── cere.yml
│   ├── devnet.yml
│   ├── mainnet.yml
│   ├── qanet.yml
│   └── testnet.yml
├── hosts
│   └── cere.yml
├── roles
│   ├── docker_setup
│   │   ├── defaults
│   │   │   └── main.yml
│   │   ├── files
│   │   ├── handlers
│   │   │   └── main.yml
│   │   ├── meta
│   │   │   └── main.yml
│   │   ├── README.md
│   │   ├── tasks
│   │   │   └── main.yml
│   │   ├── templates
│   │   ├── tests
│   │   │   ├── inventory
│   │   │   └── test.yml
│   │   └── vars
│   │       └── main.yml
│   ├── ufw
│   │   ├── defaults
│   │   │   └── main.yml
│   │   ├── files
│   │   ├── handlers
│   │   │   └── main.yml
│   │   ├── meta
│   │   │   └── main.yml
│   │   ├── README.md
│   │   ├── tasks
│   │   │   └── main.yml
│   │   ├── templates
│   │   │   └── after.rules.j2
│   │   ├── tests
│   │   │   ├── inventory
│   │   │   └── test.yml
│   │   └── vars
│   │       └── main.yml
│   └── validator_watch_tg
│       ├── defaults
│       │   └── main.yml
│       ├── files
│       │   ├── requirements.txt
│       │   ├── watch_cerelog.py
│       │   ├── watch_cerelog.service
│       │   ├── watch_syslog.py
│       │   └── watch_syslog.service
│       ├── handlers
│       │   └── main.yml
│       ├── meta
│       │   └── main.yml
│       ├── README.md
│       ├── tasks
│       │   └── main.yml
│       ├── templates
│       │   └── telegram.env.j2
│       ├── tests
│       │   ├── inventory
│       │   └── test.yml
│       └── vars
│           └── main.yml
├── scripts
│   ├── check_sync_progress.sh
│   ├── generate-session-key.sh
│   └── launch_validator_node.sh
└── vault
    └── secrets.yml
```

## servers.yml example

```yaml
all:
  hosts:
  vars:
    default:
      pkgs: [vim, mlocate, jq]
  children:
    cere:
      children:
        prep:
          hosts:
            nuclearwinter2:
              ansible_host: "192.168.0.153"
              ansible_port: 99
        mainnet:
          hosts:
            nuclearwinter:
              ansible_host: "66.11.222.333"
              ansible_port: 99
        devnet:
        qanet:
```


