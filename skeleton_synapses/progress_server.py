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

class ProgressServer(HTTPServer):
    """
    Simple http server that can be polled to get the current progress of the synapse detector tool.
    This server is passive -- the synapse detector tool must periodically 
    update the progress state by calling update_progress().
    """

    @classmethod
    def create_and_start(cls, hostname, port, disable_server_logging=True):
        """
        Start the progress server in a different thread, and return the server object.
        To stop the server, simply call its shutdown() method.
        
        disable_server_logging: If true, disable the normal HttpServer logging of every request.
        """
        server = ProgressServer( disable_server_logging, (hostname, port), ProgressRequestHandler )
        server_thread = threading.Thread( target=server.serve_forever )
        server_thread.daemon = True
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
        HTTPServer.shutdown(self)
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

class ProgressRequestHandler(BaseHTTPRequestHandler):
    """
    Request handler for the ProgressServer.  (See above for details.)
    """
    
    def do_GET(self):
        if self.path == "/detector_progress":
            self._do_get_progress()
        else:
            self.send_error( httplib.BAD_REQUEST, "Bad query syntax: {}".format( self.path ) )
    
    def _do_get_progress(self):
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
    

# quick test
if __name__ == "__main__":
    import time
    try:
        progress_server = ProgressServer.create_and_start( "localhost", 8000 )
        time.sleep(60)
    finally:
        progress_server.shutdown()
