import os

print(os.path.exists(r'dataset\\dataset_128\\imgs_color\\ILSVRC2012_val_00008010.JPEG'))
print('ILSVRC2012_val_00008010.JPEG' in os.listdir(r'dataset\\dataset_128\\imgs_gray'))
