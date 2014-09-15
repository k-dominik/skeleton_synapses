import json
import collections
import threading
import httplib
from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler

ProgressInfo = collections.namedtuple("ProgressInfo", ["node_overall_index", # overall progress
                                                       "skeleton_node_count",
                                                       "branch_index",
                                                       "skeleton_branch_count",
                                                       "node_index_in_branch", # progress within the current branch
                                                       "branch_node_count",
                                                       "total_detections"] )

class ProgressRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        with self.server._lock:
            progress = self.server.progress
        json_text = json.dumps( progress._asdict() )
        self.send_response(httplib.OK)
        self.send_header("Content-type", "text/json")
        self.send_header("Content-length", str(len(json_text)))
        self.end_headers()
        self.wfile.write( json_text )

    def log_request(self, *args, **kwargs):
        """
        Override from BaseHTTPRequestHandler, so we can respect 
          the ProgressServer's disable_logging setting.
        """
        if not self.server.disable_logging:
            BaseHTTPRequestHandler.log_request(self, *args, **kwargs )
    
class ProgressServer(HTTPServer):

    @classmethod
    def create_and_start(cls, hostname, port, disable_server_logging=True):
        """
        Start the progress server in a different thread, and return the server object.
        To stop the server, simply call its shutdown() method.
        
        disable_server_logging: If true, disable the normal HttpServer logging of every request.
        """
        server = ProgressServer( disable_server_logging, (hostname, port), ProgressRequestHandler )
        server_thread = threading.Thread( target=server.serve_forever )
        server._set_thread(server_thread)
        server_thread.start()
        return server

    def update_progress(self, progress):
        """
        Update the current progress status, which users can query via a GET request.
        """
        with self._lock:
            self.progress = progress
    
    def shutdown(self):
        """
        Stop the server and wait for its thread to finish.
        """
        HTTPServer.shutdown()
        self._shutdown_completed_event.wait()
        if self.thread:
            self.thread.join()

    def __init__(self, disable_logging, *args, **kwargs):
        """
        Constructor.  Do not call this yourself.  Instead, use create_and_start().
        """
        HTTPServer.__init__(self, *args, **kwargs)
        self.disable_logging = disable_logging
        self._shutdown_completed_event = threading.Event()
        self._lock = threading.Lock()
        self.progress = ProgressInfo(0,0,0,0,0,0,0)

    def serve_forever(self):
        """
        Override from HTTPServer.  Called for you from create_and_start()
        """
        try:
            HTTPServer.serve_forever(self)
        finally:
            self.server_close()
            self._shutdown_completed_event.set()

    def _set_thread(self, thread):
        self.thread = thread
