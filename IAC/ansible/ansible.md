# ansible
## folder structure

```bash

```

## servers.yml

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
