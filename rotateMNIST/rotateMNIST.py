
from torchvision.transforms.functional import rotate
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

class RotateMNIST(Dataset):
    def __init__(self, opt, data_type, mode):
        self.opt = opt
        self.data_type = data_type
        self.mode = mode

        if mode == 'train':
            self.hawkes_process = MHP(alpha=[[0.5]], mu=[40.0], omega=1.0)
        elif mode == 'test':
            self.hawkes_process = MHP(alpha=[[0.5]], mu=[40.0], omega=1.0)

        if opt.isLoad == True:
            # load data from pt file
            if mode == "train":
                self.data, self.labels, self.angles = torch.load('./data/load_'+mode+'_'+data_type+'.pt')
            elif mode == "test":
                self.data, self.labels, self.angles = torch.load('./data/load_'+mode+'_'+data_type+'.pt')

            print("Load "+data_type+' '+mode+"set successfully!")
        else:
            # generate from the original MNIST dataset
            if mode == "train":
                self.mnist, self.labels = torch.load(opt.train_image_path)
                num_mnist_sample = opt.num_mnist_sample_train
                self.range_angle = [0, 180]
            elif mode == "test":
                self.mnist, self.labels = torch.load(opt.test_image_path)
                num_mnist_sample = opt.num_mnist_sample_test
                self.range_angle = [180, 360]
            self.mnist = self.mnist.numpy()
            self.labels = self.labels.numpy()

            # construct nums_list that contains the set for each number
            self.nums_list = []
            self.mnist_images_list = []
            for i in range(opt.class_num):
                nums = self.mnist[self.labels == i]
                mnist_images = [Image.fromarray(self.get_picture_array(nums, r, shift=0)).resize((28, 28), Image.ANTIALIAS)
                                for r in range(nums.shape[0])]
                self.nums_list.append(nums)
                self.mnist_images_list.append(mnist_images)

            self.interval_time = 1.
            self.video_length = opt.video_length
            self.label_length = opt.label_length

            w, h = opt.canvas_size
            self.data = np.zeros((num_mnist_sample, self.video_length, w, h), dtype=np.uint8)
            self.labels = np.zeros((num_mnist_sample, self.label_length), dtype=np.long)
            self.angles = np.zeros((num_mnist_sample, self.video_length), dtype=np.float32)
            
            # generate videos 
            for idx in range(num_mnist_sample):
                img_list, label_list, angle_seq = self.generate_MNIST_single_video()
                video_array = self.imageListToNumpy(img_list)
                label_array = np.array(label_list)
                angle_array = np.array(angle_seq)

                self.data[idx] = video_array
                self.labels[idx] = label_array
                self.angles[idx] = angle_array

            self.data = torch.from_numpy(self.data)
            self.labels = torch.from_numpy(self.labels)
            self.angles = torch.from_numpy(self.angles)

            self.save('./data/load_'+mode+'_'+data_type+'.pt')
            print("Generate and save "+data_type+"  "+mode+"set successfully!")


    def save(self, path):
        torch.save((self.data, self.labels, self.angles), path)

    def load_mnist(self, image_path, label_path):
        import gzip
        with gzip.open(image_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28).transpose(0, 1, 3, 2)
        data = data / np.float32(255)
        with gzip.open(label_path, 'rb') as f:
            label = np.frombuffer(f.read(), np.uint8, offset=8)
        return data, label

    def imageListToNumpy(self, img_list, greyscale=True):
        num_images = len(img_list)
        w, h = img_list[0].size
        if greyscale:
            result = np.zeros((num_images, w, h), dtype=np.uint8)
            for i, image in enumerate(img_list):
                img_array = np.asarray(
                    image.getdata(), dtype=np.uint8).reshape(w, h)
                result[i] = img_array
        return result

    def generate_MNIST_single_video(self):
        w = 10.0
        if self.data_type == 'hks':
            time_seq = []
            while len(time_seq) < self.video_length:
                time_seq = self.hawkes_process.generate_seq(float(self.range_angle[0]/w), float(self.range_angle[1]/w))
            time_seq = [t[0] for t in time_seq]
            angle_seq = [w*t for t in time_seq]
            angle_seq = angle_seq[:self.video_length]

        elif self.data_type == 'exp':
            interval = 0.57
            time_seq = [np.exp(interval*i) for i in range(self.video_length)]            
            angle_seq = [t*w+self.range_angle[0]+np.random.randn()*5 for t in time_seq]

        img_list = []
        label_list = []
        # random select one num from [0, 9]
        num = np.random.randint(0, 9)
        # random select one image from the nums_list 
        random_int = np.random.randint(0, len(self.mnist_images_list[num])-1)
        image = self.mnist_images_list[num][random_int]

        # sample initial position and angle 
        last_position = self.sample_initial_positions()
        for _, angle in enumerate(angle_seq):
            # generate one frame 
            img_section = self.generate_one_frame(image, init_pos=last_position, init_angle=angle,
                                                    canvas_size=self.opt.canvas_size)
            
            img_list += img_section
            label_list += [num]
       
        return img_list, label_list, angle_seq

    def sample_initial_positions(self):
        position_mean, position_var = self.opt.positions
        p_x = np.random.normal(loc=position_mean[0], scale=position_var[0])
        p_y = np.random.normal(loc=position_mean[1], scale=position_var[1])
        position = (int(round(p_x)), int(round(p_y)))

        return position

    def generate_one_frame(self, image, init_pos=(18, 18), init_angle=0, canvas_size=(64, 64)):
        generated_frame_list = []
        img = self.generate_MNIST_frame(image, init_angle, init_pos, shape=canvas_size)
        generated_frame_list.append(img)
        
        return generated_frame_list

    def generate_MNIST_frame(self, image, angle, position, shape=(64, 64), isSave=False):
        width, height = shape
        canvas = Image.new('L', (width, height))
        img = rotate(image, angle)
        canvas.paste(img, position)
        if isSave:
            canvas.save(str(round(angle))+'.jpg')
        return canvas

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index].float()/255.

        return data.unsqueeze(1), self.labels[index], self.angles[index]

    @staticmethod
    def imageScale(image, scaling_factor=1.0):
        w, h = image.size
        w_modified = int(np.ceil(w*scaling_factor))
        h_modified = int(np.ceil(h*scaling_factor))
        w_modified = max(w_modified, 3)
        h_modified = max(h_modified, 3)
        scaled_image = image.resize((w_modified, h_modified))
        return scaled_image

    @staticmethod
    def arr_from_img(im, shift=0):
        w, h = im.size
        arr = im.getdata()
        c = int(np.product(arr.size) / (w*h))
        return np.asarray(arr, dtype=np.float32).reshape((h, w, c)).transpose(2, 1, 0) / 255. - shift

    @staticmethod
    def get_picture_array(X, index, shift=0):
        if X.shape[0] <= 3:
            ch, w, h = X.shape[1], X.shape[2], X.shape[3]
            ret = ((X[index]+shift)*255.).reshape(ch, w,
                                                  h).transpose(2, 1, 0).clip(0, 255).astype(np.uint8)
            if ch == 1:
                ret = ret.reshape(h, w)
        else:
            # no channel
            w, h = X.shape[1], X.shape[2]
            # ret = ((X[index]+shift)*255.).reshape(w,
            #   h).clip(0, 255).astype(np.uint8)
            ret = X[index]
        return ret

# Adapted from https://github.com/stmorse/hawkes
class MHP:
    def __init__(self, alpha=[[0.5]], mu=[0.1], omega=1.0):
        '''params should be of form:
        alpha: numpy.array((u,u)), mu: numpy.array((,u)), omega: float'''

        self.data = []
        self.alpha, self.mu, self.omega = np.array(alpha), np.array(mu), omega
        self.dim = self.mu.shape[0]
        self.check_stability()

    def check_stability(self):
        ''' check stability of process (max alpha eigenvalue < 1)'''
        w, v = np.linalg.eig(self.alpha)
        me = np.amax(np.abs(w))
        # print('Max eigenvalue: %1.5f' % me)
        if me >= 1.:
            print('(WARNING) Unstable.')

    def generate_seq(self, min, max):
        '''Generate a sequence based on mu, alpha, omega values. 
        Uses Ogata's thinning method, with some speedups, noted below'''

        self.data = []  # clear history

        Istar = np.sum(self.mu)
        s = 0.
        while s < 0.01:
            s = np.random.exponential(scale=1./Istar)
        s = np.random.exponential(scale=1./Istar)

        # attribute (weighted random sample, since sum(mu)==Istar)
        n0 = np.random.choice(np.arange(self.dim),
                              1,
                              p=(self.mu / Istar))
        self.data.append([s, n0])

        # value of \lambda(t_k) where k is most recent event
        # starts with just the base rate
        lastrates = self.mu.copy()

        decIstar = False
        while True:
            tj, uj = self.data[-1][0], int(self.data[-1][1])

            if decIstar:
                # if last event was rejected, decrease Istar
                Istar = np.sum(rates)
                decIstar = False
            else:
                # otherwise, we just had an event, so recalc Istar (inclusive of last event)
                Istar = np.sum(lastrates) + \
                    self.omega * np.sum(self.alpha[:, uj])

            # generate new event
            w = 0.
            while w < 0.01:
                w = np.random.exponential(scale=1./Istar)
            s += w

            # calc rates at time s (use trick to take advantage of rates at last event)
            rates = self.mu + np.exp(-self.omega * (s - tj)) * \
                (self.alpha[:, uj].flatten() *
                 self.omega + lastrates - self.mu)

            # attribution/rejection test
            # handle attribution and thinning in one step as weighted random sample
            diff = Istar - np.sum(rates)
            try:
                n0 = np.random.choice(np.arange(self.dim+1), 1,
                                      p=(np.append(rates, diff) / Istar))
            except ValueError:
                # by construction this should not happen
                print('Probabilities do not sum to one.')
                self.data = np.array(self.data)
                return self.data

            if n0 < self.dim:
                self.data.append([s, n0])
                # update lastrates
                lastrates = rates.copy()
            else:
                decIstar = True

            if s >= max:
                self.data = np.array(self.data)
                self.data = self.data[self.data[:, 0] < max]
                self.data = self.data[self.data[:, 0] > min] 
                return self.data
