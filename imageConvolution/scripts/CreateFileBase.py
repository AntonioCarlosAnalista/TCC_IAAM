# Criar uma estrutura de arquivos
# imagens/nomeImagemTarget/target.png --> Imagem que deve ser reconhecida pelo modelo treinado
# imagens/outras/todasAsImagensNaoTarget.png --> Outras imagens para comparação
#
# Criar as distorções do target
# Criar as distorções de outras
#
#target=Image.__new__
#others=Image.__new__
#from pathlib import Path
import os
import shutil

from sqlalchemy import null
#from LoadImagesBaseNames import ImagesBaseNames

CV='/'

class FileBase:
    #def __init__(self, target, others):
    #    self.target = target
    #    self.others = others
    #
    # Vai criar o diretório com base no local de execução
    def createDir(self, directory, d_stru):
        #img_target_directory = os.path.realpath(\
        #    os.path.join(os.path.dirname(__file__), '..', 'images', directory))
        img_target_directory = os.path.join(d_stru.imagens, directory)

        try:
            shutil.rmtree(img_target_directory)
        except:
            null
            
        os.mkdir(img_target_directory)
        return img_target_directory


    def createTargetDir(self, imbn, d_stru):
        return self.createDir(imbn.imageTarget.replace(" ",""), d_stru)

    def createOthersDir(self, d_stru):
        return self.createDir('outras',d_stru)

    def copyImageToTarget(self, target, image):
        shutil.copy(image, target)

    def copyImagesToOthers(self,others_target,images):
        #Copiando apenas a primeira imagem
        shutil.copy(images[0], others_target)
        #for im in images:
        #    shutil.copy(im, others_target)
