from struct import pack, unpack
import socket, sys, os, subprocess, time, signal
import cPickle as pickle
import SocketServer

def send_object(sock, object):
    data = pickle.dumps(object)
    header = pack('>Q', len(data))
    sock.send(header+data)

def receive_object(sock):
    header = sock.recv(8)
    length, = unpack( '>Q', header[:8] )
    data = sock.recv(length)
    object = pickle.loads(data)
    return object

class PileServerHandler(SocketServer.BaseRequestHandler):
    def handle(self):
        object = receive_object(self.request)
        send_object(self.request, object)

class PileServer:

    def inform(self, host, port):
        fn = '/tmp/snuffleconnector-%i' % os.getpid()
        oldmask = os.umask(0077)
        f = open(fn, 'w')
        f.write('%s:%i\n' % (host, port))
        f.close()
        os.umask(oldmask)

    def __init__(self, pile):
        host, port = 'localhost', 50000
        for i in range(200):
            try:
                server = SocketServer.TCPServer((host, port), PileServerHandler)
            except socket.error:
                port += 1
                continue
                
            break
            
        self.inform(host,port)
            
        server.serve_forever()

class RemotePile:
    def __init__(self, host, port, process=None):
        self.process = process
        self.port = port
        self.host = host
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        self.sock = sock
    
    def comm(self, object):
        send_object(self.sock, object)
        return receive_object(self.sock)
    
    def close(self):
        self.sock.close()
        if self.process is not None:
            if self.process.poll() is None:
                os.kill(self.process.pid, signal.SIGTERM)
                
def spawn_remote_pile(args):
    process = subprocess.Popen(args, close_fds=True)
    
    for i in range(200):
        try:        
            f = open('/tmp/snuffleconnector-%i' % process.pid, 'r')
            toks = f.read().strip().split(':')
            f.close()
        except IOError:
            time.sleep(0.01)
            continue
            
        break
    
    host, port = toks[0], int(toks[1])
    remote = RemotePile(host, port, process)
    
    return remote
    
