import shutil
from typing import List
from os import listdir, path
from os.path import isfile, join, isdir
from DataStructureInitialize import data_structure_initialize
from LoadImagesBaseNames import ImagesBaseNames
from CreateFileBase import FileBase
from CropImage import CropWithPil
from ImageAugmentation import Augmentation
from TrainImageModel import TrainImage

targetClass=''
targetClassType=''
othersClass=''

class StructureInitialize():

    def __init__(self, target_class, others_class):
        self.targetClass = target_class[:target_class.index(".")]
        self.targetClassType = target_class[target_class.index("."):]
        self.othersClass = others_class[:others_class.index(".")]

    def initialize(self, im):
        ttv=data_structure_initialize(self.targetClass, self.othersClass)
        d_stru=ttv.getDataSetStructureVO()

        imbn=ImagesBaseNames()
        imbn.imageTarget=self.targetClass
        imbn.imageOthers=self.othersClass
        imbn.imageType=self.targetClassType
        imbn.loadImagesNameAndPath(d_stru)

        cfb=FileBase()
        target = cfb.createTargetDir(imbn, d_stru)
        cfb.copyImageToTarget(target, join(d_stru.imagens_base, imbn.imageTarget+imbn.imageType))
        
        others=cfb.createOthersDir(d_stru)
        cfb.copyImagesToOthers(others, imbn.imagesOthersPathList)

        d_stru.num_img_train=5000
        d_stru.num_img_valid=2500
        d_stru.image_width = 160
        d_stru.image_heigth = 160
        d_stru.division_factor = 40
        d_stru.resize_image=160
        d_stru.perc_train = 60

        aug=Augmentation()
        aug.do_augmentation(d_stru,imbn)

        def listFilesFrom(pathToFiles):
            return [f for f in listdir(pathToFiles) if isfile(join(pathToFiles, f))]
        
        def unsort_listdir(files_list: List):
            unsorted_list=[]
            for f in files_list[:]:
                unsorted_list.append(f)
                files_list.remove(f)
            return unsorted_list

        def moveFileTo(orig, dest):
            shutil.move(orig, dest)

        # DISTRIBUIR ENTRE TREINO E VALIDAÇÃO
        #
        def distribution(d_st):
            files_from_folder = unsort_listdir(listFilesFrom(d_st.target_class_dir))
            tam_train = int(len(files_from_folder) * (d_st.perc_train / 100))
            count_train=0
            for fl in files_from_folder:
                if tam_train > count_train:
                    moveFileTo(join(d_stru.target_class_dir, fl), join(d_stru.train_target_dir,fl))
                    count_train+=1
                else:
                    moveFileTo(join(d_stru.target_class_dir, fl), join(d_stru.validation_target_dir, fl))
                    count_train+=1

            files_from_folder = unsort_listdir(listFilesFrom(d_stru.others_class_dir))
            tam_train=int(len(files_from_folder) * (d_st.perc_train /100))
            count_train=0
            for fl in files_from_folder:
                if tam_train > count_train:
                    moveFileTo(join(d_stru.others_class_dir, fl), join(d_stru.train_others_dir, fl))
                    count_train+=1
                else:
                    moveFileTo(join(d_stru.others_class_dir, fl), join(d_stru.validation_others_dir, fl))
                    count_train+=1

        distribution(d_st=d_stru)

        d_stru.target_class=imbn.imageTarget.replace(' ','')
        d_stru.others_class='outras'

        model_name=path.join(d_stru.modelos,d_stru.target_class)
        if not isdir(model_name):
            # Treinar o modelo caso ainda não exista
            trained_model = TrainImage()

            #Treinamento do modelo COMENTADO
            trained_model.executar(d_stru=d_stru)

        cwp=CropWithPil(im)
        cwp.main(d_stru.resize_image, d_stru.division_factor, d_stru=d_stru)

