import uuid

def unique_id(length=6):
    return uuid.uuid4().hex[:length].upper()


def fullname(o):
    klass = o.__class__
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__ # avoid outputs like 'builtins.str'
    return module + '.' + klass.__qualname__
