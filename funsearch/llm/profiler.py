class Profiler:
    def __init__(self):
        self._profiler = None

    # state を持つ関数にすれば 回数をカウントしたりできる
    def profile_events(self):
        ...


profile_events = Profiler().profile_events
