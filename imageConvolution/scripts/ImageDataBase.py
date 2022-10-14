from ImageAugmentation import Augmentation

class CreateImageDataBase():

    def createFiles(self, folder_origin, folder_target, img_target):
        aug=Augmentation()
        #folder_origin='C:\\ACMS\\Estudos\\NOTEBOOK\\TCC\\imagens_base', folder_target='C:\\ACMS\\Estudos\\NOTEBOOK\\imageConvolution\\images\\', img_target='Velocidade maxima permitida 50 R-19'
        # aug.do_augmentation(folder_origin=folder_origin, folder_target=folder_target, img_target=img_target)
