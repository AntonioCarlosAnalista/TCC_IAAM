import os
import shutil

from DatasetStructure import DatasetStructureVO
ds_vo=''

class data_structure_initialize():
    def __init__(self, targetClass, othersClass):
        self.ds_vo=DatasetStructureVO()
        self.ds_vo.root=os.getcwd()
        self.ds_vo.imagens_base=os.path.join(self.ds_vo.root, 'TCC','imagens_base')
        self.ds_vo.modelos = os.path.join(self.ds_vo.root, 'imageConvolution', 'modelos')
        
        self.ds_vo.imagens = os.path.join(self.ds_vo.root, 'imageConvolution', 'images')
        self.ds_vo.dataset_train = os.path.join(self.ds_vo.root, 'imageConvolution', 'images', 'train')
        self.ds_vo.dataset_validation = os.path.join(self.ds_vo.root, 'imageConvolution', 'images', 'validation')

        if os.path.exists(self.ds_vo.dataset_train) and os.path.isdir(self.ds_vo.dataset_train):
            shutil.rmtree(self.ds_vo.dataset_train)
        os.mkdir(self.ds_vo.dataset_train)

        if os.path.exists(self.ds_vo.dataset_validation) and os.path.isdir(self.ds_vo.dataset_validation):
            shutil.rmtree(self.ds_vo.dataset_validation)
        os.mkdir(self.ds_vo.dataset_validation)

        self.ds_vo.train_target_dir = os.path.join(self.ds_vo.dataset_train,targetClass.replace(' ',''))
        self.ds_vo.train_others_dir = os.path.join(self.ds_vo.dataset_train,othersClass)

        self.ds_vo.validation_target_dir = os.path.join(self.ds_vo.dataset_validation,targetClass.replace(' ',''))
        self.ds_vo.validation_others_dir = os.path.join(self.ds_vo.dataset_validation,othersClass)

        if os.path.exists(self.ds_vo.train_target_dir) and os.path.isdir(self.ds_vo.train_target_dir):
            shutil.rmtree(self.ds_vo.train_target_dir)
        os.mkdir(self.ds_vo.train_target_dir)

        if os.path.exists(self.ds_vo.train_others_dir) and os.path.isdir(self.ds_vo.train_others_dir):
            shutil.rmtree(self.ds_vo.train_others_dir)
        os.mkdir(self.ds_vo.train_others_dir)

        if os.path.exists(self.ds_vo.validation_target_dir) and os.path.isdir(self.ds_vo.validation_target_dir):
            shutil.rmtree(self.ds_vo.validation_target_dir)
        os.mkdir(self.ds_vo.validation_target_dir)

        if os.path.exists(self.ds_vo.validation_others_dir) and os.path.isdir(self.ds_vo.validation_others_dir):
            shutil.rmtree(self.ds_vo.validation_others_dir)
        os.mkdir(self.ds_vo.validation_others_dir)

        self.ds_vo.target_class_dir = os.path.join(self.ds_vo.root, 'imageConvolution', 'images', targetClass.replace(' ',''))
        self.ds_vo.others_class_dir = os.path.join(self.ds_vo.root, 'imageConvolution', 'images', 'outras') 
        # othersClass.replace(' ',''))

    def getDataSetStructureVO(self):
        return self.ds_vo
