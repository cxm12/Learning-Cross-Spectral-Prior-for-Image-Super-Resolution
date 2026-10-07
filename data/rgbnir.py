from data import srdata


class RGBNIR(srdata.SRData):
    def __init__(self, args, train=True):
        # data_range = [r.split('-') for r in args.data_range.split('/')]
        # if train:
        #     data_range = data_range[0]
        # else:
        #     if args.test_only and len(data_range) == 1:
        #         data_range = data_range[0]
        #     else:
        #         data_range = data_range[1]

        # self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(RGBNIR, self).__init__(
            args, train=train
        )
