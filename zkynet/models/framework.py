class Function:
    """
    A Function is a callable thing.
    """
    def __init__(self):
        # stores meta information about the function
        self._meta = {}
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
