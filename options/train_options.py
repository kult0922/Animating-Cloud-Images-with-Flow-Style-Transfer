from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser.add_argument('--epochs', type=int,
                help='number of end epochs', default=20)
        BaseOptions.initialize(self, parser)

        self.isTrain = True
        return parser
