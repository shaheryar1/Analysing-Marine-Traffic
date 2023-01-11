class Config:
    MYSQL_DATABASE_USER = 'admin'
    MYSQL_DATABASE_PASSWORD = 'admin'
    MYSQL_DATABASE_DB = 'ships'
    MYSQL_DATABASE_HOST = 'localhost'
    MYSQL_DATABASE_PORT = 3306
    MYSQL_CURSORCLASS = 'DictCursor'

ANNOTATIONS_PATH='Marrine_Vessel_Annotations'

CLASSES=['Container ship','Fishing vessels','Military ship','Passenger ship','Sailing vessel','Tanker','Tug']


DATASET_PATH="/home/shaheryar/Desktop/Projects/Marrine-Vessel-Detection/Dataset/Annotated_Dataset/All_Classes_Train";



labels_to_name= {
    1:'Tanker',
    2:'Tug',
    3:'Container ship',
    4:'Fishing vessels',
    5:'Passenger ship',
    6:'Sailing vessel',
    7:'Military ship',
    8:'Supply utility vessel',
    9:'Power boat',
    10:'Jet ski',
    11:'yatch'
}




def convertToBinaryData(filename):
    with open(filename, 'rb') as file:
        binaryData = file.read()
    return binaryData

