import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape, UpSampling2D, Flatten, Dense, BatchNormalization, LeakyReLU, ReLU
from tensorflow.keras.optimizers import Adam
from loss import dice_coef, dice_coef_loss
from tensorflow.keras import backend as K
from BasicConvLSTMCell import ConvLSTMCell
from tensorflow.keras.initializers import glorot_normal, Constant

from discriminator_new import Discriminator


def ResnetBlock(x, dim, ksize):
    net1 = Conv2D(dim, (ksize, ksize), activation='relu', padding='same',
                  kernel_initializer=glorot_normal(seed=None), bias_initializer=Constant(value=0))(x)
    net2 = Conv2D(dim, (ksize, ksize), activation=None, padding='same',
                  kernel_initializer=glorot_normal(seed=None), bias_initializer=Constant(value=0))(net1)
    return net2 + x
    

def ResnetBlock_r(x, dim, ksize):
    net1 = Conv2D(dim, (ksize, ksize), padding='same',
                  kernel_initializer=glorot_normal(seed=None), bias_initializer=Constant(value=0))(x)
    net1 = BatchNormalization()(net1)
    #net1 = Dropout(0.02)(net1)
    net1 = ReLU()(net1)
    #net1 = Activation('relu')(net1)
    net2 = Conv2D(dim, (ksize, ksize), activation=None, padding='same',
                  kernel_initializer=glorot_normal(seed=None), bias_initializer=Constant(value=0))(net1)
    net2 = BatchNormalization()(net2)
    #net2 = Dropout(0.02)(net2)
    return net2 + x


def make_trainable(model, val):
    model.trainable = val
    try:
        for l in model.layers:
            try:
                for k in l.layers:
                    make_trainable(k, val)
            except:
                pass
            l.trainable = val
    except:
        pass


class S_R_A_Net(object):
    def __init__(self, img_rows, img_cols):
        self.img_rows = img_rows
        self.img_cols = img_cols

        self.MaskDiscriminator = None  # Mask discriminator
        self.D_model = None

        self.Decomposer = None  # Decomposer
        self.Reconstructor = None  # Reconstructor

        self.Generator = None

        self.G_supervised_model = None  # Supervised generator trainer

    def build(self):
        self.build_discriminator_trainer()

        self.Decomposer = self.build_decomposer_trainer()
        self.Reconstructor = self.build_reconstructor_trainer()

        self.Generator = self.build_generator_trainer()

        self.build_supervised_trainer()

    def build_discriminator_trainer(self):
        D_built = Discriminator(self.img_rows, self.img_cols)
        D_built.build()
        # Mask Discriminator
        self.MaskDiscriminator = D_built.mask_discriminator

        Image_for_real = Input((self.img_rows, self.img_cols, 1))
        real_M = Input((self.img_rows, self.img_cols, 1))
        dis_real_M = self.MaskDiscriminator([Image_for_real, real_M])

        Image_for_fake = Input((self.img_rows, self.img_cols, 1))
        fake_M = Input((self.img_rows, self.img_cols, 1))
        dis_fake_M = self.MaskDiscriminator([Image_for_fake, fake_M])

        self.D_model = Model(inputs=[Image_for_real, real_M, Image_for_fake, fake_M], outputs=[dis_real_M, dis_fake_M])

        self.D_model.compile(optimizer=Adam(lr=1e-5, beta_1=0.5), loss='mse')

    def build_decomposer_trainer(self):
        inputs_list = []
        inputs_assistant = Input((self.img_rows/4, self.img_cols/4, 1))
        inputs1 = Input((self.img_rows/4, self.img_cols/4, 1))
        inputs2 = Input((self.img_rows/2, self.img_cols/2, 1))
        inputs3 = Input((self.img_rows, self.img_cols, 1))
        inputs_list.extend([inputs1, inputs2, inputs3])

        inputs_rnn = Conv2D(512, (5, 5), activation=None, padding='same', strides=(32, 32))(inputs3)
        cell = ConvLSTMCell([3, 3], 256)

        for i in range(3):
            input = inputs_list[i]
            if i == 0:
                rnn_state = K.zeros_like(inputs_rnn)
                input_all = input
            else:
                inp_preds = K.resize_images(inp_preds, 2, 2, data_format='channels_last', interpolation='bilinear')
                rnn_state = K.resize_images(rnn_state, 2, 2, data_format='channels_last', interpolation='bilinear')
                input_all = input

            conv1_1 = Conv2D(32, (5, 5), activation='relu', padding='same', kernel_initializer=glorot_normal(seed=None),
                             bias_initializer=Constant(value=0))(input_all)
            conv1_2 = ResnetBlock(conv1_1, 32, 5)
            conv1_3 = ResnetBlock(conv1_2, 32, 5)
            conv1_4 = ResnetBlock(conv1_3, 32, 5)

            conv2_1 = Conv2D(64, (5, 5), activation='relu', padding='same', strides=(2, 2),
                             kernel_initializer=glorot_normal(seed=None), bias_initializer=Constant(value=0))(conv1_4)
            conv2_2 = ResnetBlock(conv2_1, 64, 5)
            conv2_3 = ResnetBlock(conv2_2, 64, 5)
            conv2_4 = ResnetBlock(conv2_3, 64, 5)

            conv3_1 = Conv2D(128, (5, 5), activation='relu', padding='same', strides=(2, 2),
                             kernel_initializer=glorot_normal(seed=None), bias_initializer=Constant(value=0))(conv2_4)
            conv3_2 = ResnetBlock(conv3_1, 128, 5)
            conv3_3 = ResnetBlock(conv3_2, 128, 5)
            conv3_4 = ResnetBlock(conv3_3, 128, 5)

            conv4_1 = Conv2D(256, (5, 5), activation='relu', padding='same', strides=(2, 2),
                             kernel_initializer=glorot_normal(seed=None), bias_initializer=Constant(value=0))(conv3_4)
            conv4_2 = ResnetBlock(conv4_1, 256, 5)
            conv4_3 = ResnetBlock(conv4_2, 256, 5)
            conv4_4 = ResnetBlock(conv4_3, 256, 5)

            deconv1_1, rnn_state = cell(conv4_4, rnn_state)

            deconv1_2 = ResnetBlock(deconv1_1, 256, 5)
            deconv1_3 = ResnetBlock(deconv1_2, 256, 5)
            deconv1_4 = ResnetBlock(deconv1_3, 256, 5)

            deconv2_1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same',
                                        kernel_initializer=glorot_normal(seed=None),
                                        bias_initializer=Constant(value=0))(deconv1_4)

            cat1 = deconv2_1 + conv3_4
            deconv2_2 = ResnetBlock(cat1, 128, 5)
            deconv2_3 = ResnetBlock(deconv2_2, 128, 5)
            deconv2_4 = ResnetBlock(deconv2_3, 128, 5)

            deconv3_1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same',
                                        kernel_initializer=glorot_normal(seed=None),
                                        bias_initializer=Constant(value=0))(deconv2_4)

            cat2 = deconv3_1 + conv2_4
            deconv3_2 = ResnetBlock(cat2, 64, 5)
            deconv3_3 = ResnetBlock(deconv3_2, 64, 5)
            deconv3_4 = ResnetBlock(deconv3_3, 64, 5)

            deconv4_1 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same',
                                        kernel_initializer=glorot_normal(seed=None),
                                        bias_initializer=Constant(value=0))(deconv3_4)

            cat3 = deconv4_1 + conv1_4
            deconv4_2 = ResnetBlock(cat3, 32, 5)
            deconv4_3 = ResnetBlock(deconv4_2, 32, 5)
            deconv4_4 = ResnetBlock(deconv4_3, 32, 5)

            # feature integration
            if i == 2:
                modality_z = Conv2D(128, (5, 5), strides=1, padding='same')(conv4_4)
                modality_z = BatchNormalization()(modality_z)
                modality_z = ReLU()(modality_z)
                modality_z = Conv2D(64, (5, 5), strides=1, padding='same')(modality_z)
                modality_z = BatchNormalization()(modality_z)
                modality_z = ReLU()(modality_z)
                modality_z = Flatten()(modality_z)
                modality_z = Dense(32)(modality_z)
                modality_z = ReLU()(modality_z)
                modality_z = Dense(16, activation='sigmoid')(modality_z)

                fi_1 = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer=glorot_normal(seed=None),
                              bias_initializer=Constant(value=0))(deconv2_4)
                fi_1 = UpSampling2D(size=(2, 2))(fi_1)
                fi_2 = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer=glorot_normal(seed=None),
                              bias_initializer=Constant(value=0))(deconv3_4)
                fi_2 = K.concatenate([fi_2, fi_1], axis=3)
                fi_2 = UpSampling2D(size=(2, 2))(fi_2)
                inp_preds = K.concatenate([deconv4_4, fi_2], axis=3)
                inp_preds = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer=glorot_normal(seed=None),
                                   bias_initializer=Constant(value=0))(inp_preds)
            else:
                inp_preds = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer=glorot_normal(seed=None),
                                   bias_initializer=Constant(value=0))(deconv4_4)

        decomposer_model = Model(inputs=[inputs_assistant, inputs1, inputs2, inputs3], outputs=[inp_preds, modality_z])

        return decomposer_model

    def build_reconstructor_trainer(self):
        mask_input = Input((self.img_rows, self.img_cols, 1))
        resd_input = Input((16,))
        modality = Dense(32)(resd_input)
        modality = ReLU()(modality)
        modality = Dense(self.img_rows * self.img_cols)(modality)
        modality = ReLU()(modality)
        modality = Reshape((int(self.img_rows / 4), int(self.img_cols / 4), 16))(modality)
        modality = UpSampling2D(size=2)(modality)
        modality = Conv2D(16, 3, padding='same')(modality)
        modality = BatchNormalization()(modality)
        modality = ReLU()(modality)
        modality = UpSampling2D(size=2)(modality)
        modality = Conv2D(8, 3, padding='same')(modality)
        modality = BatchNormalization()(modality)
        modality = ReLU()(modality)

        conc_lr = K.concatenate([mask_input, modality])
        re_modality = ResnetBlock_r(conc_lr, 9, 3)
        re_modality = ResnetBlock_r(re_modality, 9, 3)
        re_modality = ResnetBlock_r(re_modality, 9, 3)

        re_modality = Conv2D(1, (7, 7), activation='tanh', padding='same')(re_modality)

        reconstructor_model = Model(inputs=[mask_input, resd_input], outputs=re_modality)

        return reconstructor_model

    def build_generator_trainer(self):
        # Decomposition/Segmentation
        real_assistant = Input((self.img_rows / 4, self.img_cols / 4, 1))
        real_I1 = Input((self.img_rows / 4, self.img_cols / 4, 1))
        real_I2 = Input((self.img_rows / 2, self.img_cols / 2, 1))
        real_I3 = Input((self.img_rows, self.img_cols, 1))

        fake_M, fake_Z = self.Decomposer([real_assistant, real_I1, real_I2, real_I3])

        # Reconstruction
        rec_I = self.Reconstructor([fake_M, fake_Z])

        # Reconstruction using a real Mask
        real_M = Input((self.img_rows, self.img_cols, 1))
        fake_I = self.Reconstructor([real_M, fake_Z])

        generator_model = Model(inputs=[real_assistant, real_I1, real_I2, real_I3, real_M], outputs=[fake_M, fake_I, rec_I])
        generator_model.compile(optimizer=Adam(lr=1e-5, beta_1=0.5),
                                loss=[dice_coef_loss, 'mae', 'mae'],
                                loss_weights=[10, 3, 3])

        return generator_model


    def build_supervised_trainer(self):
        # Decomposition/Segmentation
        real_assistant = Input((self.img_rows / 4, self.img_cols / 4, 1))
        real_I1 = Input((self.img_rows / 4, self.img_cols / 4, 1))
        real_I2 = Input((self.img_rows / 2, self.img_cols / 2, 1))
        real_I3 = Input((self.img_rows, self.img_cols, 1))
        fake_M, fake_Z = self.Decomposer([real_assistant, real_I1, real_I2, real_I3])

        # Reconstruction
        rec_I = self.Reconstructor([fake_M, fake_Z])

        self.MaskDiscriminator.trainable = False
        real_I = Input((self.img_rows, self.img_cols, 1))
        adv_M = self.MaskDiscriminator([real_I, fake_M])

        # Decomposition/Segmentation
        real_unlabel_assistant_1 = Input((self.img_rows / 4, self.img_cols / 4, 1))
        real_unlabel_I1_1 = Input((self.img_rows / 4, self.img_cols / 4, 1))
        real_unlabel_I2_1 = Input((self.img_rows / 2, self.img_cols / 2, 1))
        real_unlabel_I3_1 = Input((self.img_rows, self.img_cols, 1))
        fake_unlabel_M_1, fake_unlabel_Z_1 = self.Decomposer(
            [real_unlabel_assistant_1, real_unlabel_I1_1, real_unlabel_I2_1, real_unlabel_I3_1])

        # Reconstruction
        rec_unlabel_I_1 = self.Reconstructor([fake_unlabel_M_1, fake_unlabel_Z_1])

        self.G_supervised_model = Model(inputs=[real_assistant, real_I1, real_I2, real_I3, real_I,
                                                real_unlabel_assistant_1, real_unlabel_I1_1, real_unlabel_I2_1, real_unlabel_I3_1],
                                        outputs=[fake_M, adv_M, rec_I, rec_unlabel_I_1])

        self.G_supervised_model.compile(optimizer=Adam(lr=1e-5, beta_1=0.5),
                                        loss=[dice_coef_loss, 'mse', 'mae', 'mae'],
                                        loss_weights=[10, 3, 3, 3])