from my_Evison.Evison import Display,show_network
from PIL import Image
from Networks import vgg_exp3 as vgg
import argparse
import torch
import os
import torchvision

parser=argparse.ArgumentParser(description="visualization of weight")
parser.add_argument("--load_checkpoint",type=str,default=None,help="Path of model.")
parser.add_argument("--show_network",type=bool,default=False,help="If show network.")
parser.add_argument("--visualized_layer",type=str,default=None,help="Visualized layer.")
parser.add_argument("--network",type=str,default="vgg11",help="Select network.")
args=parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#PATH
model_path=os.path.join(args.load_checkpoint,"net.pth")
output_feature_dir=os.path.join(args.load_checkpoint,"feature")

try:
    net_module=getattr(vgg,args.network)
    print(args.network)
except AttributeError:
    not_module=vgg.vgg11
net=net_module(num_classes=10)
net.to(device)
if args.load_checkpoint!=None:
    net.load_state_dict(torch.load(model_path))
    print("load success!")

if args.show_network==True:
    show_network(net)

if args.visualized_layer!=None:
    test_feature_set=torchvision.datasets.CIFAR10(root='./data',train=False,download=True)
    for i in range(10):
        img=test_feature_set[i][0]
        img= img.resize((224,224),Image.ANTIALIAS)
        display=Display(net,args.visualized_layer,img_size=(224,224))
        display.save(img,path=output_feature_dir,file='test'+str(i))
        img.save(os.path.join(output_feature_dir,'test'+str(i)+'.png'))


