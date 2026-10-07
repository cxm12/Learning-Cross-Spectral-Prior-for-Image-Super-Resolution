from data import srdata


class Benchmark(srdata.SRData):
    def __init__(self, args, train=False):
        super(Benchmark, self).__init__(args, train=train)


