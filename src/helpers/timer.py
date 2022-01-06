import time


class Timer:
    """
    Helper class to simplify time measuring.
    """
    def __init__(self):
        self.time_start = 0.0
    
    def set_timer(self):
        self.time_start = time.time()
        
    def stop_timer(self):
        time_stop = time.time()
        print(f"Execution time: {'{:.2f}'.format(time_stop - self.time_start)}s")
        self.time_start = 0.0
