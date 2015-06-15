class A(object):
    def __init__(self):
        self.b = "test"

    def pr(self):
        print(self.b)


def factory():
    return A()
