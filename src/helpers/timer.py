import time


class Timer:
    """
    Helper class to simplify time measuring.
    """
    def __init__(self):
        self.time_start = 0.0
    
    def set_timer(self):
        """Set time on timer object"""
        self.time_start = time.time()
        
    def stop_timer(self):
        """Stop time on timer object and display amount of time that passed and reset timer."""
        time_stop = time.time()
        print(f"Execution time: {'{:.2f}'.format(time_stop - self.time_start)}s")
        self.time_start = 0.0
