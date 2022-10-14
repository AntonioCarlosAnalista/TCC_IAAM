import os


class DatasetStructureVO:
    def __init__(self):
        self.dataset_train = os.getcwd()
        self.dataset_validation = self.dataset_train
        self.train_target_dir = self.dataset_train
        self.train_others_dir = self.dataset_train
        self.validation_target_dir = self.dataset_train
        self.validation_others_dir = self.dataset_train
        self.target_class_dir  = self.dataset_train
        self.others_class_dir = self.dataset_train
        self.modelos = ''
        self.target_class=''
        self.others_class=''
        self.imagens=''
        self.imagens_base=''

        self.dataset_dir = ''
        self.dataset_train_dir = ''
        self.dataset_validation_dir = ''
        self.dataset_train_target = ''
        self.dataset_train_target_len = 10
        self.dataset_validation_target = ''
        self.dataset_validation_target_len = 0
        self.dataset_train_others = ''
        self.dataset_trains_others_len = 0
        self.dataset_validation_others = ''
        self.dataset_validation_others_len = 0
        self.division_factor = 0
        self.resize_image=0
        self.perc_train = 0
        self.image_color_channel = 3
        self.image_color_channel_size = 255
        self.batch_size = 32
        self.epochs = 3
        self.learning_rate = 0.00005
        self.class_names = [self.target_class, self.others_class]
        self.num_img_train=0
        self.num_img_valid=0
        self.limit=0.035
