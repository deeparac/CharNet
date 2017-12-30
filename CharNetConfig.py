class CharNetConfig(object):
    def __init__(self, params=None):
        if params is None:
            self.conv_layers = [
                    [256, 7, 3],
                    [256, 7, 3],
                    [256, 3, None],
                    [256, 3, None],
                    [256, 3, None],
                    [256, 3, 3]
                ]
            self.fc_layers = [512, 10]
            self.l0 = 128
            self.alstr = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’’’/\|_@#$%ˆ&* ̃‘+-=<>()[]{}]'
            self.alphabet_size = len(self.alstr)
        else:
            self.conv_layers = params['conv_layers']
            self.fc_layers = params['fc_layers']
            self.l0 = params['l0']
            slef.alstr = params['alstr']
