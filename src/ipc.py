from struct import pack, unpack
import socket, sys, os, subprocess, time, signal, traceback
import cPickle as pickle
import SocketServer

from pyrocko import pile

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

def connector_filename(pid):
    fn = '/tmp/snuffleconnector-%s-%i' % (os.environ['USER'], pid)
    return fn

class RPCHandler(SocketServer.BaseRequestHandler):
        
    def handle(self):
        object = receive_object(self.request)
        
        response = None
        exception = None
        traceback_exc = None
        if isinstance(object, tuple) and len(object) > 0 and isinstance(object[0], str):
            try:
                response = self.server.rpcserver.call(object[0], object[1], object[2])
            except Exception, e:
                exception = e
                traceback_exc = '\n--- begin remote traceback ---\n'+traceback.format_exc()+'--- end remote traceback ---' 
        
        send_object(self.request, (response, exception, traceback_exc))

class RPCServer:    

    def __init__(self):
        self.connector_filename = None
        self.exposed = {}
        
        host, port = 'localhost', 50000
        for i in range(200):
            try:
                server = SocketServer.TCPServer((host, port), RPCHandler)
                server.rpcserver = self
            except socket.error:
                port += 1
                continue
                
            break
        
        self.server = server
        self.inform(host,port)
    
    
    def inform(self, host, port):
        fn = connector_filename(os.getpid())
        oldmask = os.umask(0077)
        f = open(fn, 'w')
        f.write('%s:%i\n' % (host, port))
        f.close()
        os.umask(oldmask)
        
        self.connector_filename = fn
        
    def run(self):
        self.server.serve_forever()
        
    def expose(self, methodname):
        self.exposed[methodname] = True
    
    def call(self, methodname, args, kwargs):
        if methodname in self.exposed:
            return getattr(self, methodname)(*args, **kwargs)
        
    def __del__(self):
        import os
        fn = self.connector_filename
        if fn is not None:
            if os.path.isfile(fn):
                os.unlink(fn)

class RemoteException(Exception):
    pass

class RPCClient:
    
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
                
    def call(self, methodname, args, kwargs):
        response, exception, traceback_exc = self.comm( (methodname, args, kwargs) )
        if exception is not None:
            raise RemoteException(traceback_exc)
        return response
        
                
    def __getattr__(self, methodname):
                    
        def remote_method_call(*args, **kwargs):
            return self.call(methodname, args, kwargs)
        return remote_method_call
            

class RemotePileServer(RPCServer):
    
    def __init__(self, pile):
        RPCServer.__init__(self)
        self.pile = pile
        self.expose('add_trace')
        
    def add_trace(self, tr):
        memfile = pile.MemTracesFile(None,[tr])
        self.pile.add_file(memfile)
        

class RemotePile(RPCClient):
    pass
    

class ConnectionFailed(Exception):
    pass

class ConnectorFilePermissionsTooOpen(Exception):
    pass

def spawn_remote_pile(args):
    process = subprocess.Popen(args, close_fds=True)
    
    toks = None
    for i in range(200):
        try:
            fn = connector_filename(process.pid)
            f = open(fn, 'r')
            mode = os.fstat(f.fileno())[0]
            if mode & 077 != 0:
                f.close()
                raise ConnectorFilePermissionsTooOpen(fn)
            
            toks = f.read().strip().split(':')
            f.close()
            os.unlink(fn)
            
        except IOError:
            time.sleep(0.01)
            continue
            
        break
    
    if toks is None:
        raise ConnectionFailed()
    
    host, port = toks[0], int(toks[1])
    remote = RemotePile(host, port, process)
    
    return remote
    
