from __future__ import division

import json
import collections
import threading
import httplib
from datetime import datetime, timedelta
import time
from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler

from skeleton_synapses.constants import MONITOR_HOST, MONITOR_PORT, MONITOR_INTERVAL

MAX_POLLING_INTERVAL = 0.1

ProgressInfo = collections.namedtuple("ProgressInfo", [
    "name",
    "description",
    "time_started",
    "time_elapsed",
    "time_to_go",
    "time_at_finish",
    "time_updated",
    "items_total",
    "items_done",
    "items_remaining",
    "proportion_done",
])


class DummyThread(object):
    def start(self):
        pass

    def stop(self):
        pass


class QueueMonitorThread(threading.Thread):
    def __init__(
            self, queue, hostname=MONITOR_HOST, port=MONITOR_PORT, items_total=None, name='', description='',
            interval=MONITOR_INTERVAL
    ):
        super(QueueMonitorThread, self).__init__(name=name + ' thread')
        self.monitor_name = name
        self.description = description

        self.time_started = None
        self.queue = queue
        self.given_items_total = items_total
        self.items_total = None

        self.hostname = hostname
        self.port = port
        self.server = None

        self.interval = interval

        self._lock = threading.Lock()
        self.stop_event = threading.Event()
        self.last_polled = None
        self.polling_interval = min(self.interval, MAX_POLLING_INTERVAL)

        self.daemon = True

    def progress_from_queue(self):
        items_remaining = self.queue.qsize()
        items_done = self.items_total - items_remaining
        proportion_done = items_done / self.items_total

        time_updated = datetime.now()
        time_elapsed = time_updated - self.time_started
        if proportion_done:
            time_to_go = timedelta(seconds=time_elapsed.seconds / proportion_done - time_elapsed.seconds)
            time_at_finish = time_updated + time_to_go
        else:
            time_to_go = 'Inf'
            time_at_finish = 'Inf'

        return ProgressInfo(
            self.monitor_name, self.description,
            str(self.time_started), str(time_elapsed), str(time_to_go), str(time_at_finish), str(time_updated),
            self.items_total, items_done, items_remaining,
            proportion_done
        )

    def run(self):
        self.server = ProgressServer.create_and_start(self.hostname, self.port)
        self.time_started = datetime.now()
        self.last_polled = datetime.min  # ensure it polls immediately
        self.items_total = self.given_items_total or self.queue.qsize()

        while not self.queue.empty() and not self.stop_event.is_set():
            if (datetime.now() - self.last_polled).seconds >= self.interval:
                progress = self.progress_from_queue()
                self.server.update_progress(progress)
            time.sleep(self.polling_interval)

        self.server.shutdown()
        self.server = None
        self.time_started = None
        self.stop_event.clear()

    def stop(self):
        self.stop_event.set()
        self.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.join()


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
        server = ProgressServer(disable_server_logging, (hostname, port), ProgressRequestHandler)
        server_thread = threading.Thread(target=server.serve_forever)
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
        self.progress = ProgressInfo('', '', '', '', '', '', '', 0, 0, 0, 0)

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
        if self.path == "/progress":
            self._do_get_progress()
        else:
            self.send_error(httplib.BAD_REQUEST, "Bad query syntax: {}".format(self.path))

    def _do_get_progress(self):
        with self.server._lock:
            progress = self.server.progress
        json_text = json.dumps(progress._asdict(), sort_keys=True, indent=2)
        self.send_response(httplib.OK)
        self.send_header("Content-type", "text/json")
        self.send_header("Content-length", str(len(json_text)))
        self.end_headers()
        self.wfile.write(json_text)

    def log_request(self, *args, **kwargs):
        """
        Override from BaseHTTPRequestHandler, so we can respect
          the ProgressServer's disable_logging setting.
        """
        if not self.server.disable_logging:
            BaseHTTPRequestHandler.log_request(self, *args, **kwargs)


# quick test
if __name__ == "__main__":
    import time
    try:
        progress_server = ProgressServer.create_and_start("localhost", 8000)
        time.sleep(60)
    finally:
        progress_server.shutdown()
