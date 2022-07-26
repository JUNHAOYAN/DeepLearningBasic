"""
cache class, used to store cache
"""


class Cache:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __setitem__(self, key, value):
        self.kwargs[key] = value

    def __getitem__(self, item):
        if item in self.kwargs:
            return self.kwargs[item]

    def __repr__(self):
        return self.kwargs.keys()


# if __name__ == '__main__':
#     c = Cache()
#     c["a"] = 1
#     print(c["a"])
