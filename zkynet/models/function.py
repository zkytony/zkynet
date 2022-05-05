class Function:
    """
    A function maps inputs to outputs (the forward pass);
    The function also stores its own gradient functions,
    for each input.

    When the function is called for a forward pass, it
    also accumulates gradient of the output with respect
    to every input - that gradient is also a function.
    """
    def __init__(self):
        # stores meta information about the function
        self._meta = {}
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
