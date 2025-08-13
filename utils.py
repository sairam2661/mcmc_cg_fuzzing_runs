import datetime

def timestamp(millis=False):
    if millis:
        return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    else:
        return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")