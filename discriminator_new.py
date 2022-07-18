from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, LeakyReLU, ReLU, Flatten, Dense, Concatenate, BatchNormalization

class Discriminator(object):
    def __init__(self, img_rows, img_cols):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.image_discriminator = None
        self.mask_discriminator = None

    def build(self):
        self.build_mask_discriminator()

    def build_mask_discriminator(self):
        input_R = Input((self.img_rows, self.img_cols, 1))
        input_P = Input((self.img_rows, self.img_cols, 1))

        input_D = Concatenate()([input_R, input_P])

        M_D = Conv2D(32, (3, 3), strides=2, padding='same')(input_D)
        M_D = BatchNormalization()(M_D)
        #M_D = ReLU()(M_D)
        M_D = LeakyReLU(0.2)(M_D)

        M_D = Conv2D(64, (3, 3), strides=2, padding='same')(M_D)
        M_D = BatchNormalization()(M_D)
        #M_D = ReLU()(M_D)
        M_D = LeakyReLU(0.2)(M_D)

        M_D = Conv2D(128, (3, 3), strides=2, padding='same')(M_D)
        M_D = BatchNormalization()(M_D)
        #M_D = ReLU()(M_D)
        M_D = LeakyReLU(0.2)(M_D)

        M_D = Conv2D(256, (3, 3), strides=1, padding='same')(M_D)
        M_D = BatchNormalization()(M_D)
        #M_D = ReLU()(M_D)
        M_D = LeakyReLU(0.2)(M_D)

        M_D = Flatten()(M_D)
        M_D = Dense(1, activation='sigmoid')(M_D)

        self.mask_discriminator = Model(inputs=[input_R, input_P], outputs=M_D)