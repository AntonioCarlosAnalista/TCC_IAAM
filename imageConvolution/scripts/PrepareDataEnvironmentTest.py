from PrepareDataEnvironment import StructureInitialize

if __name__ =="__main__":
    #stru_ini=StructureInitialize(target_class='Velocidade maxima permitida 50 R-19.JPG', \
    #    others_class='outras')

    stru_ini=StructureInitialize(target_class='De preferencia R-2.JPG', \
        others_class='Sentido de circulacao da via ou pista R-24a.JPG')
    
    
    #stru_ini=StructureInitialize(target_class='Parada obrigatoria R1.JPG', \
    #    others_class='Sentido de circulacao da via ou pista R-24a.JPG')
    
    # Essa imagem não tem
    #stru_ini=StructureInitialize(target_class='Sentido de circulacao da via ou pista R-24a.JPG', \
    #    others_class='Conserve se a direita R-23.JPG')

    #stru_ini.initialize(im="C:\\ACMS\\Estudos\\NOTEBOOK\\imageConvolution\\images\\3980.png")
    #stru_ini.initialize(im="C:\\ACMS\\Estudos\\NOTEBOOK\\imageConvolution\\images\\3030.png")
    #stru_ini.initialize(im="C:\\ACMS\\Estudos\\NOTEBOOK\\imageConvolution\\images\\0.png")
    stru_ini.initialize(im="C:\\ACMS\\Estudos\\NOTEBOOK\\imageConvolution\\images\\4660.png")
    
