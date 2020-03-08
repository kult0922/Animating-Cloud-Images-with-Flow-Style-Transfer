from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser.add_argument('--drivingVideo', default = '../dataset/sky_cloud',
                help='path to drivenVideo')
        parser.add_argument('--sourceImage', default = '../dataset/sky_cloud',
                help='path to sourveImage')
        BaseOptions.initialize(self, parser)
        self.isTrain = False

        return parser
