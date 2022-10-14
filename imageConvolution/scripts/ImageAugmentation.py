#Visa, a partir de imagens em formato plano natural, gerar novas imagens (distorcidas) 
# com técnicas de data augmentation.

from os import listdir
from os.path import isfile, join
from os import remove
import PIL
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.image_utils import img_to_array, load_img
from LoadImagesBaseNames import ImagesBaseNames
from pathlib import Path


class Augmentation():
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        #rescale=1./255,
        #shear_range=0.2,
        zoom_range=0.8,
        # horizontal_flip=True,      # Há placas de trânsito que não podem ser invertidas.
        fill_mode='nearest',
        brightness_range=[0.2,1.0])
        
    def loadFileNames(self, imbn):
        return imbn.imageTargetPath, imbn.imagesOthersPathList
    
    def listFiles(self, pathToFiles):
        return [f for f in listdir(pathToFiles) if isfile(join(pathToFiles, f))]
    
    def resizeImage(self, img, d_stru):
        image = img.resize((d_stru.image_width, d_stru.image_heigth), PIL.Image.ANTIALIAS)
        return image

    def augmentationGenerator(self, image_origin_path, folder_target, qtd, d_stru):
        img=load_img(image_origin_path)

        img=self.resizeImage(img, d_stru)

        img_shape = img_to_array(img)                          # Numpy array com shape (3, 150, 150) 
        img_shape = img_shape.reshape((1,) + img_shape.shape)  # Numpy array com shape (1, 3, 150, 150)

        nome_base_img=image_origin_path[image_origin_path.rfind('\\')+1:int(image_origin_path.rfind('.'))]
        i = 0

        # O comando .flow() abaixo gera imagens transformadas randomicamente e salva o resultado
        for batch in self.datagen.flow(\
            img_shape, batch_size=1, save_to_dir=folder_target,\
                save_prefix=nome_base_img, save_format='jpeg'):
            i += 1
            if i >= qtd:
                break  # Para evitar loop infinito
        file_to_remove=join(folder_target, nome_base_img+'.JPG')
        remove(file_to_remove)
        return nome_base_img
        
    def make_train_validation():
        null

    def do_augmentation(self, d_stru, imbn):
        folder_target=d_stru.imagens
        # Carrega as paths completas dos arquivos na origem
        image_target_path, image_target_others = self.loadFileNames(imbn)

        # Gera novas imagens (augmentation) a partir da imagem target obtidas e salva no destino
        # Gera N imagens target
        self.augmentationGenerator(\
            image_target_path, join(\
                folder_target,imbn.imageTarget.replace(' ','')), d_stru.num_img_train, d_stru)
            
        # Gera novas imagens (augmentation) a partir das imagens others obtidas e salva no destino de outras
        # Gera 20 imagens para cada uma das outras placas existentes
        # for others in image_target_others:
        #    self.augmentationGenerator(others, folder_target+'\\'+'outras', 2000)
        # Gera apenas 1 imagem das outras placas existentes
        self.augmentationGenerator(\
            image_target_others[0], join(\
                folder_target,'outras'), d_stru.num_img_valid, d_stru)
