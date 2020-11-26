import os
import random 

def get_img_label_paths(images_path, labels_path):
    res = []
    for dir_entry in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, dir_entry)):
            file_name, _ = os.path.splitext(dir_entry)
            res.append((os.path.join(images_path, file_name+".tif"),
                        os.path.join(labels_path, file_name+".png")))
    return res

def save_to_file(l, filename, save_path):
    def func(item):
        return ','.join(item) + '\n'
    with open(os.path.join(save_path, filename), 'w') as f:
        f.writelines(map(func, l))
        
def generate_train_valid(images_path, labels_path, valid_size=0.2, save_path='./'):

    res = get_img_label_paths(images_path, labels_path)
    n_valid = int(len(res) * valid_size)
    n_train = len(res) - n_valid
    random.shuffle(res)
    train = res[:n_train]
    valid = res[-n_valid:]
    
    save_to_file(train, 'train_list_253.txt', save_path)
    save_to_file(valid, 'valid_list_253.txt', save_path)
        
if __name__ == "__main__":
    path = '/media/caixh/database/RemoteSensing'
    images_path = os.path.join(path,'images')
    labels_path = os.path.join(path,'labels')
    valid_size = 0.2
    save_path = '/home/caixh/project/segmentation/baseline/data/'

    # split  
    generate_train_valid(images_path, labels_path, valid_size=valid_size, save_path=save_path)
    
    # all 
    # res = get_img_label_paths(images_path, labels_path)
    # # random.shuffle(res)
    # save_to_file(res, 'all_train_list.txt', save_path)