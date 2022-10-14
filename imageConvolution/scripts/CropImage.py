# Realizando convoluções e cortes em uma imagem fornecida tenta encontrar o modelo fornecido
#
# from turtle import width
from PIL import Image
from pip import main
from sqlalchemy import true
import tensorflow as tf
import os

image=Image.__new__
imagePath=''
count=0
acum=0

class CropWithPil:
    def __init__(self, img):
        self.imagePath = img
        self.count = 0
        self.acum = 0
        self.colision = []

    def countColisions(self, cropRetangle, side):
        if len(self.colision)==0:
            point_score=[]
            point_score.append(int(cropRetangle[0]))
            point_score.append(int(cropRetangle[1]))
            point_score.append(1)
            self.colision.append(point_score)
        else:
            found=False
            for coli in range(0,len(self.colision)):
                point_score=self.colision[coli]
                if int(cropRetangle[0]) >= (point_score[0] - int(side / 2)) and\
                    int(cropRetangle[0]) <= (point_score[0] + int(side / 2)) and\
                        int(cropRetangle[1]) >= (point_score[1] - int(side / 2)) and\
                            int(cropRetangle[1]) <= (point_score[1] + int(side / 2)):
                    point_score[2]=point_score[2]+1
                    found=True
                    break
            if not found:
                point_score=[]
                point_score.append(int(cropRetangle[0]))
                point_score.append(int(cropRetangle[1]))
                point_score.append(1)
                self.colision.append(point_score)
                found=False
    
    def predict(self, model, croped_im):

        #img = tf.keras.preprocessing.image.load_img(croped_im, target_size = d_stru.image_size)
        #img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.keras.preprocessing.image.img_to_array(croped_im)
        img = tf.expand_dims(img, 0)

        return model.predict(img, verbose=0)[0][0]

    ### Procura na imagem utilizando o modelo
    def findImage(self, model, croped_im, d_stru):
        self.count=self.count + 1

        prediction = self.predict(model, croped_im)
        #print(prediction)

        # armazena para verificação o pedaço da imagem que será comparado
        if prediction < d_stru.limit:
            self.acum = self.acum + 1
            if self.acum > 8:
                img_path='C:\\ACMS\\Estudos\\NOTEBOOK\\imageConvolution\\images\\lixo\\im_'+\
                    str(self.count)+'_'+str(prediction)+'.png'
                croped_im.save(img_path, quality=95)
                print(f'Acumulo: {self.acum}')
                print('Prediction: {0} | {1} | {2}'\
                    .format(prediction, (d_stru.target_class if prediction < d_stru.limit else d_stru.others_class),\
                        self.count))

        else:
            self.acum = 0

        return prediction

    def newSizeImage(self, im, resizeImage):
        if resizeImage > 0:
            im=im.resize((resizeImage,resizeImage), Image.ANTIALIAS)
            #im.show()
            return im
        else:
            return im
        
    # Retorna a imagem cropada e com redimensionamento definido
    def cropDefinedArea(self, crop_rectangle, im, resizeImage):
        return self.newSizeImage(im.crop(crop_rectangle), resizeImage)

    def findInPiece(self, im, pieces, resizeImage, model, d_stru):

        side = int(im.height / pieces) if im.height < im.width  else int(im.width / pieces)

        horizontal = int(im.width / side) * 4
        vertical = int(im.height / side) * 4
        seq=0
        imgs=[]

        for ind_hor in range(1, horizontal):
            #print(f'ind_hor {ind_hor}')
            im_left = im.width - (ind_hor * (side / 4))
            im_rigth = im_left + side
            for ind_ver in range(1, vertical):
                im_upper = im.height - (ind_ver * (side / 4))
                im_lower = im_upper + side
                if im_lower <= im.height:
                    crop_rectangle = (im_left, im_upper, im_rigth, im_lower)
                    cropped_im = self.cropDefinedArea(crop_rectangle, im, resizeImage)
                    if cropped_im.getbbox() is None:
                        continue
                    #self.findImage(model, cropped_im, d_stru=d_stru)
                    ### BREAK quando a cropped_im for a imagem procurada
                    prediction=self.findImage(model, cropped_im, d_stru=d_stru)
                    img_path='C:\\ACMS\\Estudos\\NOTEBOOK\\imageConvolution\\images\\lixo\\cr_'+\
                            str(crop_rectangle[0])+'X'+str(crop_rectangle[1])+\
                                '_'+str(prediction)+'.png'
                    cropped_im.save(img_path, quality=95)
                    self.count=self.count + 1
                    if prediction < d_stru.limit:
                        #print("Corte: ",crop_rectangle)
                        print(f"Encontrado: {crop_rectangle}")
                        img_path='C:\\ACMS\\Estudos\\NOTEBOOK\\imageConvolution\\images\\lixo\\im_'+\
                            str(self.count)+'_'+str(prediction)+'.png'
                        cropped_im.save(img_path, quality=95)

                        self.countColisions(cropRetangle=crop_rectangle, side=side)
                        #print(self.colision)

                        #imgs.append(img_path)
                        #seq = seq + 1
                        #if seq >= 4:
                        #    for img_seq in imgs:
                        #        print('sequencia')
                        #    seq=0
                        #    imgs=[]
                    else:
                        seq=0
                        imgs=[]
        maior=0
        crop=[0,0]
        for i in range(0, len(self.colision)):
            actual=self.colision[i][2]
            if actual > maior:
                maior=actual
                crop[0]=self.colision[i][0]
                crop[1]=self.colision[i][1]

        crop_rectangle=[crop[0]-side, crop[1]-side, crop[0]+side, crop[1]+side]
        img_path='C:\\ACMS\\Estudos\\NOTEBOOK\\imageConvolution\\images\\lixo\\_1_im_.png'

        im=im.crop(crop_rectangle)
        im.save(img_path)

        return False, None


    # Define o crop da imagem em camadas
    # Começa da direita inferior para a esquerda superior
    # Para cada deslocamento realiza redimensionamento do crop
    # Inicializa a busca de acordo com modelo treinado
    #
    #
    def imgCropAndFind(self, divisor,im, resizeImage,d_stru):
        model = tf.keras.models.load_model(os.path.join(d_stru.modelos,d_stru.target_class))

        lado =int(im.height / divisor) if im.height < im.width  else int(im.width / divisor)
        for lin in range((im.width - lado),1,-round(lado/2)):
            for col in range((im.height - lado),1,-round(lado/2)):
                for tam in range(lado, divisor, -10):
                    crop_rectangle = (lin, col, lin+tam, col+tam)
                    cropped_im = self.cropDefinedArea(crop_rectangle, im, resizeImage)
                    if cropped_im.getbbox() is None:
                        continue
                    self.findImage(model, cropped_im, d_stru=d_stru)
                    ### BREAK quando a cropped_im for a imagem procurada
                    #if self.findImage(model, cropped_im, d_stru=d_stru):
                    #    print("Corte: ",crop_rectangle)
                    #    print(f"Encontrado: {crop_rectangle}")
                    #    #return True, crop_rectangle
        return False, None

    # Iniciador do processo
    # Define o fator de divisão baseado a altura e largura da imagem
    # Retorna True ou False conforme encontrado o modelo na imagem ou não
    #
    def main(self, resizeImage, baseLength, d_stru):
        model = tf.keras.models.load_model(os.path.join(d_stru.modelos,d_stru.target_class))
        self.image=Image.open(str(self.imagePath))
        piece=(self.image.width / d_stru.image_width) + 1.5
        status, coord = self.findInPiece(self.image, 16, resizeImage=resizeImage, model=model, d_stru=d_stru)
        self.image.close()
        return True

