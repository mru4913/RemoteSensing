import os 
import glob
import zipfile
import importlib

def generate_outputs(pyfile_path, input_paths, output_dir):
    # model prediction file
    predict_py = pyfile_path+".model_predict" 
    # model definition file  
    define_py = pyfile_path+".model_define"  

    # get init_model function 
    init_model = getattr(importlib.import_module(define_py), "init_model") 
    # get predict function 
    predict = getattr(importlib.import_module(predict_py), "predict") 

    model = init_model()
    for input_path in input_paths:
        predict(model, input_path, output_dir)

def read_img_from_file(filename):
    res = []
    with open(filename, 'r') as f:
        for i in f.readlines():
            res.append(tuple(i.strip().split(','))[0])
    return res

def zip_dir(dirname, zipfilename):
    filelist = []

    for file_name in os.listdir(dirname):
        filelist.append(os.path.join(dirname, file_name))

    with zipfile.ZipFile(zipfilename, "w") as zf:
        for tar in filelist:
            arcname = tar[len(dirname):]
            print(tar + " -->rar: "+ arcname)
            zf.write(tar,arcname)

if __name__ == "__main__":

    main_path = os.path.dirname(__file__)
    user_zip = os.path.join(os.path.dirname(__file__), 'test.zip') # 后台存储的选手上传的压缩包
    img_dir = "..."
    img_file = ""
    user_dir = "user"
    output_dir = os.path.dirname(__file__) + "../test_labels"
    os.makedirs (output_dir, exist_ok=True)

    zip_dir(main_path, user_zip)
    f_size_MB = os.path.getsize(user_zip)/1024.0/1024.0
    if f_size_MB > 500:
        score = 0 # zip文件超过500得分为0
        print("Failed.")
        exit(0)

    # with zipfile.ZipFile(user_zip, 'r') as f:
    #     f.extractall('./user') 
    # img_paths = glob.glob(img_dir + '/' + '*') # 后台存储的测试集图片路径
    img_paths = read_img_from_file(img_file)

    # # 此处相当于使用了user.model_predict，因此选手需要将所有用到的文件直接打包为zip，而不是放到一个文件夹中再把文件夹压缩为zip
    # generate_outputs(user_dir, img_paths, output_dir)