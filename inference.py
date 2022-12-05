import torch
import cv2
import torchvision.transforms as transforms
import argparse
from Model import CNNModel


class Butterfly:
    def __init__(self,path = 'outputs/model.pth',numclass=100):
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_class = numclass
        self.labels = [
                        'adonis', 'american snoot', 'an 88', 'banded peacock', 'beckers white', 
                        'black hairstreak', 'cabbage white', 'chestnut', 'clodius parnassian', 
                        'clouded sulphur', 'copper tail', 'crecent', 'crimson patch', 
                        'eastern coma', 'gold banded', 'great eggfly', 'grey hairstreak', 
                        'indra swallow', 'julia', 'large marble', 'malachite', 'mangrove skipper',
                        'metalmark', 'monarch', 'morning cloak', 'orange oakleaf', 'orange tip', 
                        'orchard swallow', 'painted lady', 'paper kite', 'peacock', 'pine white',
                        'pipevine swallow', 'purple hairstreak', 'question mark', 'red admiral',
                        'red spotted purple', 'scarce swallow', 'silver spot skipper', 
                        'sixspot burnet', 'skipper', 'sootywing', 'southern dogface', 
                        'straited queen', 'two barred flasher', 'ulyses', 'viceroy', 
                        'wood satyr', 'yellow swallow tail', 'zebra long wing'
                    ]
        self.transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5]
                        )
                        ])
        self.model = self.load_model(path)
    
    def load_model(self,path):
        model = CNNModel(self.num_class).to(self.device)
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def procesing_image(self,image):
        # convert to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        # add batch dimension
        image = torch.unsqueeze(image, 0) #theo batch
        return image
        
    def predict(self,image):
        image = self.procesing_image(image)
        with torch.no_grad():
            outputs = self.model(image.to(self.device))
        output_label = torch.topk(outputs, 1) # tra ve k phan tu lon nhat: theo batch
        pred_class = self.labels[int(output_label.indices)]
        return pred_class

    @staticmethod
    def draw(orig_image,gt_class,pred_class):
        cv2.putText(orig_image, 
        f"GT: {gt_class}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.6, (0, 255, 0), 2, cv2.LINE_AA
        )
        cv2.putText(orig_image, 
            f"Pred: {pred_class}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, (0, 0, 255), 2, cv2.LINE_AA
        )
        print(f"GT: {gt_class}, pred: {pred_class}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='/media/ngocthien/DATA/Python_basic/archive/test/CHALK HILL BLUE/1.jpg',help='path to the input image')
    parser.add_argument('--num_class',default=100)
    args = vars(parser.parse_args())
    
    '''model'''
    model = Butterfly(numclass=args['num_class'])

    # read and preprocess the image
    image = cv2.imread(args['input'])
    # get the ground truth class
    gt_class = args['input'].split('/')[-2]
    orig_image = image.copy()
    pred_class = model.predict(orig_image)
    model.draw(orig_image,gt_class,pred_class)
  
    '''show result'''
    cv2.imshow('Result', orig_image)
    cv2.waitKey(0)
    cv2.imwrite(f"outputs/{gt_class}{args['input'].split('/')[-1].split('.')[0]}.png",orig_image)
    cv2.destroyAllWindows()