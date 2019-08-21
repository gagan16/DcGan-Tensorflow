from __future__ import division
import tensorflow as tf
import time
from glob import glob
import imageio
from skimage.transform import resize
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Reshape
from keras.layers import Flatten, BatchNormalization, Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam

class Dcgan(object):
    def __init__(self,sess,args):
        self.model_name = "Dcgan"  # name for checkpoint
        self.sess = sess
        self.dataset_name = args.dataset
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir=args.result_dir

        self.log_dir = args.log_dir
        self.epoch=args.epoch
        self.batch_size=args.batch_size
        self.image_size=args.img_size
        self.learning_rate=args.learning_rate
        self.print_freq=args.print_freq
        self.c_dim = 1
        self.channel=3
        self.z_dim=128
        self.image_shape=[self.image_size,self.image_size,self.channel]

        print()

        print("##### Information #####")
        print("# GAN:",self.model_name)
        print("# dataset : ", self.dataset_name)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)


        print("# Image size : ", self.image_size)
        print("# learning rate : ", self.learning_rate)

        print()


    #discriminator structure
    def construct_discriminator(self,image_shape):
        discriminator = Sequential()
        discriminator.add(Conv2D(filters=64, kernel_size=(5, 5),
                                 strides=(2, 2), padding='same',
                                 data_format='channels_last',
                                 kernel_initializer='glorot_uniform',
                                 input_shape=(image_shape)))
        discriminator.add(LeakyReLU(0.2))

        discriminator.add(Conv2D(filters=128, kernel_size=(5, 5),
                                 strides=(2, 2), padding='same',
                                 data_format='channels_last',
                                 kernel_initializer='glorot_uniform'))
        discriminator.add(BatchNormalization(momentum=0.5))
        discriminator.add(LeakyReLU(0.2))

        discriminator.add(Conv2D(filters=256, kernel_size=(5, 5),
                                 strides=(2, 2), padding='same',
                                 data_format='channels_last',
                                 kernel_initializer='glorot_uniform'))
        discriminator.add(BatchNormalization(momentum=0.5))
        discriminator.add(LeakyReLU(0.2))

        discriminator.add(Conv2D(filters=512, kernel_size=(5, 5),
                                 strides=(2, 2), padding='same',
                                 data_format='channels_last',
                                 kernel_initializer='glorot_uniform'))
        discriminator.add(BatchNormalization(momentum=0.5))
        discriminator.add(LeakyReLU(0.2))

        discriminator.add(Flatten())
        discriminator.add(Dense(1))
        discriminator.add(Activation('sigmoid'))

        optimizer = Adam(lr=0.0002, beta_1=0.5)
        discriminator.compile(loss='binary_crossentropy',
                              optimizer=optimizer,
                              metrics=None)

        return discriminator

    # Generator structure
    def construct_generator(self):
        generator = Sequential()

        generator.add(Dense(units=4 * 4 * 512,
                            kernel_initializer='glorot_uniform',
                            input_shape=(1, 1, 100)))
        generator.add(Reshape(target_shape=(4, 4, 512)))
        generator.add(BatchNormalization(momentum=0.5))
        generator.add(Activation('relu'))

        generator.add(Conv2DTranspose(filters=256, kernel_size=(5, 5),
                                      strides=(2, 2), padding='same',
                                      data_format='channels_last',
                                      kernel_initializer='glorot_uniform'))
        generator.add(BatchNormalization(momentum=0.5))
        generator.add(Activation('relu'))

        generator.add(Conv2DTranspose(filters=128, kernel_size=(5, 5),
                                      strides=(2, 2), padding='same',
                                      data_format='channels_last',
                                      kernel_initializer='glorot_uniform'))
        generator.add(BatchNormalization(momentum=0.5))
        generator.add(Activation('relu'))

        generator.add(Conv2DTranspose(filters=64, kernel_size=(5, 5),
                                      strides=(2, 2), padding='same',
                                      data_format='channels_last',
                                      kernel_initializer='glorot_uniform'))
        generator.add(BatchNormalization(momentum=0.5))
        generator.add(Activation('relu'))
        generator.add(Conv2DTranspose(filters=32, kernel_size=(5, 5),
                                      strides=(2, 2), padding='same',
                                      data_format='channels_last',
                                      kernel_initializer='glorot_uniform'))
        generator.add(BatchNormalization(momentum=0.5))
        generator.add(Activation('relu'))

        generator.add(Conv2DTranspose(filters=3, kernel_size=(5, 5),
                                      strides=(2, 2), padding='same',
                                      data_format='channels_last',
                                      kernel_initializer='glorot_uniform'))
        generator.add(Activation('tanh'))

        optimizer = Adam(lr=0.00015, beta_1=0.5)
        generator.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=None)

        return generator

    def build_model(self):

        generator = self.construct_generator()
        # generator structure
        generator.summary()
        discriminator = self.construct_discriminator(self.image_shape)

        #discriminator structure
        discriminator.summary()
        gan = Sequential()
        # Only false for the adversarial model
        discriminator.trainable = False
        gan.add(generator)
        gan.add(discriminator)

        optimizer = Adam(lr=0.00015, beta_1=0.5)
        gan.compile(loss='binary_crossentropy', optimizer=optimizer,
                    metrics=None)

        #Loading the dataset and converting images to feed to discrminator
        path = glob('dataset/'+self.dataset_name)
        batch = np.random.choice(path, self.batch_size)
        imgs = []
        for img in batch:
            img = self.imread(img)
            img = resize(img, self.image_shape)
            imgs.append(img)

        number_of_batches = int(len(path) / self.batch_size)

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        #saving checkpoints
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / number_of_batches)
            start_batch_id = checkpoint_counter - start_epoch * number_of_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS ",counter)
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")


        # Variables that will be used to plot the losses from the discriminator and
        # the adversarial models
        adversarial_loss = np.empty(shape=1)
        discriminator_loss = np.empty(shape=1)
        batches = np.empty(shape=1)

        current_batch = 0

        # Let's train the DCGAN for n epochs
        for epoch in range(start_epoch,self.epoch):

            print("Epoch " + str(epoch + 1) + "/" + str(self.epoch) + " :")

            for batch_number in range(start_batch_id,number_of_batches):

                start_time = time.time()

                # Get the current batch and normalize the images between -1 and 1
                real_images = np.array(imgs) / 127.5 - 1.

                # The last batch is smaller than the other ones, so we need to
                # take that into account
                current_batch_size = real_images.shape[0]

                # Generate noise
                noise = np.random.normal(0, 1,
                                         size=(current_batch_size,) + (1, 1, 100))

                # Generate images
                generated_images = generator.predict(noise)

                # Add some noise to the labels that will be
                # fed to the discriminator
                real_y = (np.ones(current_batch_size) -
                          np.random.random_sample(current_batch_size) * 0.2)
                fake_y = np.random.random_sample(current_batch_size) * 0.2

                # Let's train the discriminator
                discriminator.trainable = True

                d_loss = discriminator.train_on_batch(real_images, real_y)
                d_losss= discriminator.train_on_batch(generated_images, fake_y)
                d_loss =d_loss+ d_losss

                discriminator_loss = np.append(discriminator_loss, d_loss)
                losses = np.empty(shape=1)
                losses = np.append(losses, d_loss)
                losses=np.append(losses,d_losss)


                # Now it's time to train the generator
                discriminator.trainable = False

                noise = np.random.normal(0, 1,
                                         size=(current_batch_size * 2,) +
                                              (1, 1, 100))

                # We try to mislead the discriminator by giving the opposite labels
                fake_y = (np.ones(current_batch_size * 2) -
                          np.random.random_sample(current_batch_size * 2) * 0.2)

                g_loss = gan.train_on_batch(noise, fake_y)
                losses=np.append(losses,g_loss)

                adversarial_loss = np.append(adversarial_loss, g_loss)
                batches = np.append(batches, current_batch)

                # Each 5 batches show and save images
                if ((batch_number + 1) % 5 == 0 and
                        current_batch_size == self.batch_size):
                    # save_generated_images(generated_images, epoch, batch_number)
                    self.sample_images(generated_images, epoch, batch_number)

                self.write_to_tensorboard(batch_number, self.writer, losses)

                time_elapsed = time.time() - start_time
                counter += 1
                # Display and plot the results
                print("     Batch " + str(batch_number + 1) + "/" +
                      str(number_of_batches) +
                      " generator loss | discriminator loss : " +
                      str(g_loss) + " | " + str(d_loss) + ' - batch took ' +
                      str(time_elapsed) + ' s.')

                current_batch += 1

            # Save the model weights each 5 epochs
            if (epoch + 1) % 5 == 0:
                discriminator.trainable = True
                generator.save('models/generator_epoch' + str(epoch) + '.hdf5')
                discriminator.save('models/discriminator_epoch' +
                                   str(epoch) + '.hdf5')

            # Each epoch update the loss graphs
            # plt.figure(1)
            # plt.plot(batches, adversarial_loss, color='green',
            #          label='Generator Loss')
            # plt.plot(batches, discriminator_loss, color='blue',
            #          label='Discriminator Loss')
            # plt.title("DCGAN Train")
            # plt.xlabel("Batch Iteration")
            # plt.ylabel("Loss")
            # if epoch == 0:
            #     plt.legend()
            # plt.pause(0.0000000001)
            # plt.show()
            # plt.savefig('trainingLossPlotceleba.png')


            start_batch_id=0
            # print(counter)
            self.save(self.checkpoint_dir,counter)
            # self.visualize_results(epoch)
        print("main counter",counter)
        self.save(self.checkpoint_dir,counter)

    def imread(self,path):
        return imageio.imread(path, pilmode='RGB').astype(np.float)

    def sample_images(self,generated_images, epoch, batch_i):
        os.makedirs(self.result_dir+'/'+self.dataset_name, exist_ok=True)

        # Translate images to the other domain
        fake_B = generated_images

        #        # Rescale images 0 - 1
        fake_B = 0.5 * fake_B + 0.5

        imageio.imwrite(self.result_dir+'/'+self.dataset_name+'/%d_Fake%d.png' % (epoch, batch_i), fake_B[0])

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def write_to_tensorboard(self,generator_step, summary_writer,
                             losses):

        summary = tf.Summary()

        value = summary.value.add()
        value.simple_value = losses[1]
        value.tag = 'Critic Real Loss'

        value = summary.value.add()
        value.simple_value = losses[2]
        value.tag = 'Critic Fake Loss'

        value = summary.value.add()
        value.simple_value = losses[3]
        value.tag = 'Generator Loss'

        value = summary.value.add()
        value.simple_value = losses[1] - losses[2]
        value.tag = 'Critic Loss (D_real - D_fake)'

        value = summary.value.add()
        value.simple_value = losses[1] + losses[2]
        value.tag = 'Critic Loss (D_fake + D_real)'

        summary_writer.add_summary(summary, generator_step)
        summary_writer.flush()

