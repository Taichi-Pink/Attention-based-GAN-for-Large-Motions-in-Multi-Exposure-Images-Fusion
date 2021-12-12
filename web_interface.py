from flask import Flask,render_template,session, redirect,request,url_for,session
import threading
import os,json
from glob import glob
import webbrowser
app = Flask(__name__)


def launchTensorBoard(path):
    str = 'tensorboard --logdir=' + path +' --host=localhost'
    os.system(str)
    return

def run_python(test_path,file_name,train):
    os.system('python '+file_name+' --test_path='+test_path+' --train='+train)
    return

def train_python(train_path,file_name,train):
    os.system('python '+file_name+' --train_path='+train_path+' --train='+train)
    return

@app.route('/')
def index():
    return render_template('display.html')

@app.route('/display',methods=['GET','POST'])
def display():
    if request.method=='GET':
        print('display get\n')
        return render_template('display.html')
    else:
        print('display post\n')
        test_path=request.form.get('path')
        model = request.form.get('model')
        save_path = r'F:\Graduation\code\test_result'
        if test_path!=None and model!=None:

            if model=='cgan':
                prefix = "cgan_"
                file_name = 'conditional_gan.py'
            elif model=='wgan':
                prefix = "wgan_"
                file_name = 'wgan_result.py'
            else:
                prefix = "wgan_attention_"
                file_name = 'wgan_attention.py'

            session['save_path'] = save_path
            session['test_path'] = test_path
            session['model'] = model

            LDR_list = sorted(glob(os.path.join(save_path, prefix+'*LDRs.png')))
            HDR_list = sorted(glob(os.path.join(save_path, prefix+'*tonemapped.png')))

            # run test and set results
            if(len(LDR_list)==0):
                t = threading.Thread(target=run_python, args=(test_path,file_name,'False'))
                t.start()

            # read txt file, get corresponding psnr value for each generated .hdr
            psnr_list = []
            with open(os.path.join(save_path, prefix+'psnr.txt'),'r') as f:
                while True:
                    lines = f.readline()  # 整行读取数据
                    if not lines:
                        break
                    if (lines[0] == 'b' or lines[0] == '\n'):
                        continue
                    _, E_tmp = [i for i in lines.split(':')]  # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
                    E_tmp = [i for i in E_tmp.split('\n')]
                    psnr_list.append(E_tmp[0])

                f.close()

            #set values

            LDR_list0= []
            HDR_list0 = []
            for i in range(len(LDR_list)):
                p0=os.path.split(LDR_list[i])
                p1=os.path.split(HDR_list[i])
                LDR_list0.append(p0[1])
                HDR_list0.append(p1[1])

            session['LDR_list'] = LDR_list0
            session['HDR_list'] = HDR_list0
            session['psnr_list'] = psnr_list
            session.permanent = True
            #redirect
            return redirect(url_for('display'))
        else:
            return u' Please input test path!!!'


@app.route('/getSession',methods=['GET','POST'])
def getSession():
    return_json = {}
    return_json['LDR_list'] = session.get('LDR_list')
    return_json['HDR_list'] = session.get('HDR_list')
    return_json['psnr_list'] = session.get('psnr_list')
    js = json.dumps(return_json, ensure_ascii=False)
    # print(js)
    if request.method=='GET':
        # print('-------------++++\n')
        return
    else:
        # print(js)
        # print('6666666666666666666\n')
        return js


@app.route('/train',methods=['GET','POST'])
def train():
    if request.method=='GET':
        print('train get\n')
        return render_template('train.html')
    else:
        print('train post\n')
        train_path=request.form.get('train_path')
        train_model = request.form.get('train_model')
        print(train_path)
        print(train_model)
        if train_path!=None and train_model!=None:
            train_save_path = r"F:\Graduation\code\train_weight"
            if train_model=='cgan':
                train_file_name = 'conditional_gan.py'
            elif train_model=='wgan':
                train_file_name = 'wgan_result.py'
            else:
                train_file_name = 'wgan_attention.py'

            session['train_save_path'] = train_save_path
            session['train_path'] = train_path
            session['train_model'] = train_model
            t = threading.Thread(target=train_python, args=(train_path,train_file_name,'True'))
            t.start()
            print('train.............\n')
            #redirect
            return redirect('/train')
        else:
            return u' Please input test path!!!'

@app.route('/tensorboard',methods=['GET','POST'])
def tensorboard():
    train_model = request.form.get('train_model')
    if train_model == 'cgan':
        path = r"F:\Graduation\code\log\conditional_gan"
    elif train_model == 'wgan':
        path = r"F:\Graduation\code\log\wgan_result"
    else:
        path = r"F:\Graduation\code\log\wgan_attention"

    t = threading.Thread(target=launchTensorBoard, args=([path]))
    t.start()
    webbrowser.open_new_tab("http://localhost:6006")
    return render_template('display.html')


if __name__ == '__main__':
    app.secret_key = 'djskla'
    app.debug = True
    app.run(host='127.0.0.1', port=9090)


