import argparse
import torch
from torchvision import models
from PIL import Image
import numpy as np
import json

parser=argparse.ArgumentParser(description='Testing the model on an image')
parser.add_argument("image_dir", type = str, help="The path to the image that we want to test")
parser.add_argument("checkpoint",type=str, help="The path to our model's saved checkpoint")
parser.add_argument("-tk","--top_k", type = int, default = 5, help =" The number of the highest predicted classes")
parser.add_argument("-cn","--category_names",type=str, default="cat_to_name.json", help="A JSON file that maps the class values to class names")
parser.add_argument("-g","--gpu",action="store_true", help="If an available GPU will be used")

args=parser.parse_args()

checkpoint_path=args.checkpoint
imagePath=args.image_dir
topK=args.top_k
cat_names=args.category_names
gpu=args.gpu


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model=getattr(models,checkpoint['architecture'])(pretrained=True)
    model.classifier=checkpoint["classifier"]
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx=checkpoint["mapping_classes"]
    
    return model


CheckPointmodel =load_checkpoint(checkpoint_path)

print("Checkpoint model was loaded successfully \n")



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    im=Image.open(image)
    size=266,254
    im.thumbnail(size)
    
    left=(266-224)/2
    upper= (254 - 224)/2
    right= 266 - left
    bottom = 254 - upper
    
    im_crop=im.crop((left,upper,right,bottom))
    np_image=np.array(im_crop)/255
    
    means=np.array([0.485,0.456,0.406])
    stds=np.array([0.229,0.224,0.225])
    
    np_image= (np_image - means)/stds
    np_image_transposed=np_image.transpose(2,1,0)
    output = torch.from_numpy(np_image_transposed).type(torch.FloatTensor)
    return output


def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    probabilities=[]
    class_number=[]
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("GPU will be utilised, if available \n")
    else:
        device=torch.device('cpu')
        
    model.to(device)
    
    model.eval()
    
    image=process_image(image_path)
    image=image.unsqueeze_(0)
    image=image.to(device)
    
    ps=torch.exp(model(image))
    
    top_p,top_class = ps.topk(topk,dim=1)
    
    mydict=model.class_to_idx
    inv_dict={v:k for k,v in mydict.items()}
    
    Numpy_Top_class=top_class.cpu().numpy()
    Numpy_TopP=top_p.cpu().detach().numpy()
    
    List_Top_Class=Numpy_Top_class.tolist()
    List_TopP=Numpy_TopP.tolist()
    
    for i in List_Top_Class[0]:
        class_number.append(inv_dict[i])
        
    for i in List_TopP[0]:
        probabilities.append(i)
        
    
    return probabilities,class_number


probs, class_numbers = predict(imagePath,CheckPointmodel,topK)

with open(cat_names, 'r') as f:
    cat_to_name = json.load(f)

categories=[]

for i in class_numbers:
    categories.append(cat_to_name[i])

print("Top", topK, "predicted results in order: \n")
print(probs)
print(categories)