model1 = [nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True), ]
        model1 += [nn.ReLU(True), ]
        model1 += [norm_layer(64), ]

        model2 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True), ]
        model2 += [nn.ReLU(True), ]
        model2 += [norm_layer(128), ]

        model3 = [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True), ]
        model3 += [nn.ReLU(True), ]
        model3 += [norm_layer(256), ]

        model4 = [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), ]
        model4 += [nn.ReLU(True), ]
        model4 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), ]
        model4 += [nn.ReLU(True), ]

        model4 += [nn.Conv2d(256, num_bins, kernel_size=1, stride=1, padding=0, bias=True), ]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)

        self.softmax = nn.LogSoftmax(dim=1)
        if loss == 'RarityWeighted':
            self.model5 = nn.Upsample(scale_factor=4, mode='bilinear')
        elif loss == 'L2':
            self.model5 = nn.Sequential(
                nn.Conv2d(num_bins, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False),
                nn.Upsample(scale_factor=4, mode='bilinear'))