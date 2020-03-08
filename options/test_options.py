from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser.add_argument('--drivingVideo', default = '../dataset/sky_cloud',
                help='path to drivenVideo dir')
        parser.add_argument('--sourceImage', default = '../dataset/sky_cloud',
                help='path to source image')
        parser.add_argument('--mode', default = 'transfer',
                help='test mode (transfer or reconstruction)')
        BaseOptions.initialize(self, parser)
        self.isTrain = False

        return parser
