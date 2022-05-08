import uuid

def unique_id(length=6):
    return uuid.uuid4().hex[:length].upper()
