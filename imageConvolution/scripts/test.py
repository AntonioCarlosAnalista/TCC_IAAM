import os
from DatasetStructure import DatasetStructureVO
ds_vo=''
ds_vo=DatasetStructureVO()
ds_vo.root=os.getcwd()

print(ds_vo.root)