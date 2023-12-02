# docker logs to telegram

## container log watch

Identifies a docker container, recognizes patterns, keeps watch over the constantly changing file, sends notifications to telegram application

```python
#!/usr/bin/python3
import json
import os
from pathlib import Path
import re
import time
import traceback
import logging
import urllib3
import requests
import socket
import docker
import inotify.adapters
import inotify.calls
from dotenv import load_dotenv

LOG = logging.getLogger(__name__)

load_dotenv(dotenv_path="/home/powrusr/cere/envs/telegram.env", encoding="utf-8")
CERE_CONTAINER_NAME = "cere"  # add_validation_node_custom (default name)
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
node = socket.gethostname()
tg_prefix = f"{node} => {CERE_CONTAINER_NAME} container ==> "


class DockerUtils:

    def __init__(self):
        self.client = None
        self.api_client = None

    def initiate_remote_docker_connection(self, docker_server_url):
        try:
            self.client = docker.DockerClient(base_url=docker_server_url)
            self.api_client = docker.APIClient(base_url=docker_server_url)
            LOG.debug("The DockerClient version is {}".format(self.client.version()['Version']))
            self.client.ping()
        except ConnectionError:
            LOG.error('Error connecting to docker')
            raise ValueError("Could not setup Docker connection, is docker running ?")

    def initiate_local_docker_connection(self):
        try:
            self.client = docker.from_env()
            self.api_client = docker.APIClient(base_url=os.environ.get('DOCKER_HOST', 'unix://var/run/docker.sock'))
            self.client.ping()
            self.api_client.ping()
        except ConnectionError:
            LOG.error('Error connecting to docker')
            raise ValueError("Could not setup Docker connection, is docker running ?")

    def get_client(self):
        return self.client

    def get_api_client(self):
        return self.api_client

    def inspect_container(self, containerid):
        return self.api_client.inspect_container(containerid)

    def gather_container_stats(self, container_id):
        return self.api_client.stats(container=container_id, stream=False)


def escape_special_chars(text: str):
    # https://core.telegram.org/bots/api#markdownv2-style
    # https://stackoverflow.com/questions/3411771/best-way-to-replace-multiple-characters-in-a-string
    for ch in ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']:
        if ch in text:
            text = text.replace(ch, "\\" + ch)
    return text


def send_tg_message(message, parse_mode="MarkdownV2"):
    data_msg = {"chat_id": CHAT_ID,
                "text": message,
                "parse_mode": parse_mode,
                "disable_notification": False}

    data = json.dumps(data_msg)
    headers = {'Content-Type': 'application/json'}
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'

    # ?chat_id={CHAT_ID}&parse_mode={PARSE_MODE}&text={message}'
    response = requests.post(url,
                             data=data,
                             headers=headers,
                             verify=False)
    return response.json()


# .*? lazy, match as few as possible
def look_for_pattern(line):
    log_patterns = r'(.*?)warning(.*?) |' \
                   r'(.*?)warn(.*?) | ' \
                   r'(.*?)failure(.*?) | ' \
                   r'(.*?)failed(.*?) | ' \
                   r'(.*?)fail(.*?) | ' \
                   r'(.*?)error(.*?)| ' \
                   r'(.*?)emerg(.*?)| ' \
                   r'(.*?)alert(.*?)| ' \
                   r'(.*?)crit(.*?)'
    log_pattern = re.compile(log_patterns, flags=re.IGNORECASE)
    exclude_patterns = r'(.*?)\[OCW\] No local accounts available\.(.*?)'
    exclude_pattern = re.compile(exclude_patterns, flags=re.IGNORECASE)
    # match is faster than search with large files, \b becomes (.*?) as you deal with entire line using match
    if log_pattern.match(line):
        if not exclude_pattern.match(line):
            tg_ready_line = escape_special_chars(tg_prefix) + escape_special_chars(line)
            tg_ready = tg_ready_line.strip("\n")
            send_tg_message(f"{tg_ready}")
            # print(f"##################### {tg_ready}")


def chown_log_file(log):
    os.chmod(log, 0o644)


def process(line, history=False):
    if history:
        print('=', line.strip('\n'))
    else:
        print('>', line.strip('\n'))


# you need to run as root to open up a docker container log file!
def start_watch(logfile):
    from_beginning = True
    notifier = inotify.adapters.Inotify()
    while True:
        try:
            # ------------------------- check
            if not Path(logfile).exists():
                print('logfile does not exist')
                time.sleep(1)
                continue
            print('opening and starting to watch', logfile)
            # ------------------------- open
            file = open(logfile, 'r')
            if from_beginning:
                for line in file.readlines():
                    # process(line, history=True)
                    look_for_pattern(line)
            else:
                file.seek(0, 2)
                from_beginning = True
            # ------------------------- watch
            notifier.add_watch(logfile)
            try:
                for event in notifier.event_gen():
                    if event is not None:
                        (header, type_names, watch_path, filename) = event
                        if set(type_names) & set(['IN_MOVE_SELF']):  # moved
                            print('logfile moved')
                            notifier.remove_watch(logfile)
                            file.close()
                            time.sleep(1)
                            break
                        elif set(type_names) & set(['IN_MODIFY']):  # modified
                            for line in file.readlines():
                                # process(line, history=False)
                                look_for_pattern(line)
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                notifier.remove_watch(logfile)
                file.close()
                time.sleep(1)
            # -------------------------
        except (KeyboardInterrupt, SystemExit):
            break
        except inotify.calls.InotifyError:
            time.sleep(1)
        except IOError:
            time.sleep(1)
        except:
            traceback.print_exc()
            time.sleep(1)


du = DockerUtils()
du.initiate_local_docker_connection()

# get cere container stats
container_stats_cere = du.gather_container_stats(container_id=CERE_CONTAINER_NAME)
container_id_cere = container_stats_cere["id"]

# get cere container details
cere_data = du.inspect_container(containerid=CERE_CONTAINER_NAME)

# use cere_data dictionary
cere_long_id = cere_data["Id"]
cere_online = bool(cere_data["State"]["Running"])
cere_log = cere_data["LogPath"]
cere_running_image = cere_data["Image"]

start_watch(cere_log)
```

### ensure script launch on startup

service file configuration for systemd

```bash
[Unit]
Description=Cere container log watcher
After=docker.service

[Service]
ExecStart=/usr/bin/python3 /usr/local/bin/watch_cerelog.py
Restart=on-failure
RestartSec=5


[Install]
WantedBy=default.target
```

## watch host system log files

```python
#!/usr/bin/python3
import os
import time
import traceback
# import threading
import urllib3
import requests
import re
import json
import inotify.adapters
import inotify.calls
import socket
from dotenv import load_dotenv

# chmod 0644 file.py
# cp file.py /usr/local/bin/
# python3.8 -m pip install python-dotenv inotify urllib3 requests

log_file_to_watch = '/var/log/syslog'

load_dotenv(dotenv_path="/home/dadude/cere/envs/watchlogstelegram.env", encoding="utf-8")
# load_dotenv(dotenv_path="files/watchlogstelegram.env")
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
node = socket.gethostname()
tg_prefix = f"{node}: syslog ==> "


def escape_special_chars(text: str):
    # https://core.telegram.org/bots/api#markdownv2-style
    # https://stackoverflow.com/questions/3411771/best-way-to-replace-multiple-characters-in-a-string
    for ch in ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']:
        if ch in text:
            text = text.replace(ch, "\\" + ch)
    return text


def send_tg_message(message, parse_mode="MarkdownV2"):
    data_msg = {"chat_id": CHAT_ID,
                "text": message,
                "parse_mode": parse_mode,
                "disable_notification": False}

    data = json.dumps(data_msg)
    headers = {'Content-Type': 'application/json'}
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'

    # ?chat_id={CHAT_ID}&parse_mode={PARSE_MODE}&text={message}'
    response = requests.post(url,
                             data=data,
                             headers=headers,
                             verify=False)
    return response.json()


def look_for_pattern(line):
    log_patterns = r'(.*?)warning(.*?) |' \
                   r'(.*?)warn(.*?) | ' \
                   r'(.*?)failure(.*?) | ' \
                   r'(.*?)failed(.*?) | ' \
                   r'(.*?)fail(.*?) | ' \
                   r'(.*?)error(.*?)'
    log_pattern = re.compile(log_patterns, flags=re.IGNORECASE)
    exclude_patterns = r'(.*?)got error while decoding json(.*?) |' \
                       r'(.*?)ansible.legacy.command: Invoked with _raw_params(.*?)'
    exclude_pattern = re.compile(exclude_patterns, flags=re.IGNORECASE)

    # match is faster than search with large files, \b becomes (.*?) as you deal with entire line using match
    if log_pattern.match(line):
        if not exclude_pattern.match(line):
            tg_ready_line = escape_special_chars(tg_prefix) + escape_special_chars(line)
            tg_ready = tg_ready_line.strip("\n")
            send_tg_message(f"{tg_ready}")


def process(line, history=False):
    if history:
        print('=', line.strip('\n'))
    else:
        print('>', line.strip('\n'))


# https://www.linode.com/docs/guides/monitor-filesystem-events-with-pyinotify/
# https://stackoverflow.com/questions/44407834/detect-log-file-rotation-while-watching-log-file-for-modification
def start_watch(logfile):
    from_beginning = True
    notifier = inotify.adapters.Inotify()
    while True:
        try:
            # ------------------------- check
            if not os.path.exists(logfile):
                print('logfile does not exist')
                time.sleep(1)
                continue
            print('opening and starting to watch', logfile)
            # ------------------------- open
            file = open(logfile, 'r')
            if from_beginning:
                for line in file.readlines():
                    # process(line, history=True)
                    look_for_pattern(line)
            else:
                file.seek(0, 2)
                from_beginning = True
            # ------------------------- watch
            notifier.add_watch(logfile)
            try:
                for event in notifier.event_gen():
                    if event is not None:
                        (header, type_names, watch_path, filename) = event
                        if set(type_names) & set(['IN_MOVE_SELF']):  # moved
                            print('logfile moved')
                            notifier.remove_watch(logfile)
                            file.close()
                            time.sleep(1)
                            break
                        elif set(type_names) & set(['IN_MODIFY']):  # modified
                            for line in file.readlines():
                                # process(line, history=False)
                                look_for_pattern(line)
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                notifier.remove_watch(logfile)
                file.close()
                time.sleep(1)
            # -------------------------
        except (KeyboardInterrupt, SystemExit):
            break
        except inotify.calls.InotifyError:
            time.sleep(1)
        except IOError:
            time.sleep(1)
        except:
            traceback.print_exc()
            time.sleep(1)


start_watch(log_file_to_watch)
```

### service file

```bash
[Unit]
Description=host syslog tg watcher
After=network.service

[Service]
ExecStart=/usr/bin/python3 /usr/local/bin/watch_syslog.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```
