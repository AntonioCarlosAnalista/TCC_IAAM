# Trata de ler a pasta com as imagens base
# Seleciona a imagem target
# Seleciona as imagens não target
import os

DV='\\'
imageTarget=''
imageOthers=''
imageType=''
imageTargetPath=''
imagesOthersPath=[]


class ImagesBaseNames():
    def __init__(self):
        self.imageTarget=''
        self.imageOthers=''
        self.imageType=''
        self.imageTargetPath=''
        self.imagesOthersPathList=[]
        
    def loadImagesNameAndPath(self, d_stru):
        for folder, subs, files in os.walk(d_stru.imagens_base):
            for file in files:
                if file.find(self.imageTarget)!=-1:
                    arq=os.path.join(folder, file)
                    self.imageTargetPath=arq
                else:
                    if file.find(self.imageOthers)!=-1:
                        arq=os.path.abspath(os.path.join(folder, file))
                        self.imagesOthersPathList.append(arq)


