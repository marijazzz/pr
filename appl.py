#libraries
import sys
from PyQt5.QtWidgets import (QMainWindow, QTextEdit,
    QAction, QFileDialog, QApplication, QMessageBox, QPushButton)
from PyQt5.QtGui import QIcon
from skimage import filters
from skimage.segmentation import watershed
from skimage import io as skio
from skimage.transform import resize
import numpy as np
import model_cl
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import torch
import torchvision.transforms as tt
from PIL import Image
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from glob import glob
import os


#datasetclass
class FruitTestDataset(Dataset):
    def __init__(self, path, class_names, transform=tt.ToTensor()):
        self.class_names = class_names
        self.data = np.array(glob(os.path.join(path, '*.jpg')))
        self.transform=transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = Image.open(self.data[idx])
        name = self.data[idx].split('/')[-2]
        y = self.class_names.index(name)
        img = self.transform(img)
            
        return img, y
    

#plotting class
class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=100, height=100, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

#app itself
class Example(QMainWindow):
#initialisation
    def __init__(self):
        super().__init__()

        self.initUI()


    def initUI(self):

        self.statusBar()

        openFile = QAction(QIcon('open.png'), 'Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open new File')
        openFile.triggered.connect(self.showDialog)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFile)
        self.setGeometry(300, 300, 600, 600)
        self.setWindowTitle('File dialog')
        self.show()
#image processing
    def process_image(self, path):
        
        img = skio.imread(path, as_gray=True)
        
        img1 = skio.imread(path)

        sobel = filters.sobel(img)
        blurred = filters.gaussian(sobel, sigma=2.0)
        seed_mask = np.zeros(img.shape, dtype=np.int)
        seed_mask[0, 0] = 1 # backgroun
        seed_mask[600, 400] = 2 # foreground
        ws = watershed(blurred, seed_mask)
        for i in range(len(ws)):
            for j in range(len(ws[i])):
                if ws[i][j] == 1:
                    img1[i, j, :] = 255 
        img1 = resize(img1, (100, 100))
        path = path.split('/')
        p = ''
        for i in range(len(path) - 1):
            p += path[i]
            p += '/'
        path1 = os.path.join(p, 'test')
        os.mkdir(path1)
        fig=plt.figure()
        fig.set_size_inches(1, 1)
        ax=fig.add_subplot(1,1,1)
        plt.axis('off')
        plt.imshow(img1)
        plt.savefig(p + 'test/5.jpg', dpi=100)
        
        
        
        sc = MplCanvas(self, width=5, height=4, dpi=100)
        sc.axes.imshow(img1)
        self.setCentralWidget(sc)
        return img1
#some pipeline work
    def showDialog(self):

        fname = QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        image = self.process_image(fname)
        stats = ((0.68401635, 0.5785742,  0.50372875),
        (0.3034052, 0.3599607, 0.39141408))
        test_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])
        fname = fname.split('/')
        path = ''
        for i in range(len(fname) - 1):
            path += fname[i]
            path += '/'
        path += 'test/'
        print(path)
        test_dataset = FruitTestDataset(path, ['test'], transform=test_tfms)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
        self.predict(test_loader)
        path = path[:len(path)-1]
        os.rmdir(path)
        
#making predictions
    def predict(self, test_loader):
        model = model_cl.Net(131)
        print('here')
        model.load_state_dict(torch.load('model.pickle'))
        with torch.no_grad():
            for images, labels in test_loader:
                output = model(images)
        top5prob, top5label = torch.topk(output, 5)
        top5prob = top5prob.detach().numpy()
        top5label = top5label.detach().numpy()
        res_labels = []
        res_proba = []
        classes = {0: 'Quince', 1: 'Grapefruit White', 2: 'Granadilla', 3: 'Orange', 4: 'Apple Red 3', 5: 'Grape White 2', 6: 'Corn Husk', 7: 'Tamarillo', 8: 'Banana Red', 9: 'Nectarine Flat', 10: 'Pepper Yellow', 11: 'Nut Forest', 12: 'Pear Monster', 13: 'Fig', 14: 'Tomato Heart', 15: 'Onion Red Peeled', 16: 'Lemon Meyer', 17: 'Onion Red', 18: 'Passion Fruit', 19: 'Cucumber Ripe', 20: 'Cactus fruit', 21: 'Tomato not Ripened', 22: 'Mango Red', 23: 'Apple Pink Lady', 24: 'Pomegranate', 25: 'Plum', 26: 'Pineapple', 27: 'Tomato 1', 28: 'Cherry 2', 29: 'Apple Red 2', 30: 'Avocado ripe', 31: 'Dates', 32: 'Maracuja', 33: 'Papaya', 34: 'Nut Pecan', 35: 'Pear Stone', 36: 'Cherry Wax Yellow', 37: 'Eggplant', 38: 'Apple Golden 2', 39: 'Guava', 40: 'Beetroot', 41: 'Tomato Maroon', 42: 'Potato Red', 43: 'Apple Red Delicious', 44: 'Cherry Wax Red', 45: 'Kiwi', 46: 'Cherry Wax Black', 47: 'Limes', 48: 'Cantaloupe 2', 49: 'Apple Braeburn', 50: 'Pear', 51: 'Carambula', 52: 'Tomato 3', 53: 'Onion White', 54: 'Cherry 1', 55: 'Strawberry', 56: 'Lychee', 57: 'Redcurrant', 58: 'Rambutan', 59: 'Potato Red Washed', 60: 'Tomato 4', 61: 'Hazelnut', 62: 'Tomato Yellow', 63: 'Plum 3', 64: 'Grape White', 65: 'Pineapple Mini', 66: 'Mulberry', 67: 'Grape Blue', 68: 'Pear Abate', 69: 'Melon Piel de Sapo', 70: 'Pepper Orange', 71: 'Cauliflower', 72: 'Nectarine', 73: 'Salak', 74: 'Cocos', 75: 'Chestnut', 76: 'Blueberry', 77: 'Apple Granny Smith', 78: 'Banana Lady Finger', 79: 'Apricot', 80: 'Walnut', 81: 'Apple Crimson Snow', 82: 'Grapefruit Pink', 83: 'Tangelo', 84: 'Peach Flat', 85: 'Pear Forelle', 86: 'Pepper Red', 87: 'Tomato Cherry Red', 88: 'Pear Williams', 89: 'Clementine', 90: 'Apple Golden 3', 91: 'Apple Red 1', 92: 'Pear 2', 93: 'Plum 2', 94: 'Cantaloupe 1', 95: 'Lemon', 96: 'Physalis with Husk', 97: 'Peach 2', 98: 'Pepino', 99: 'Huckleberry', 100: 'Potato White', 101: 'Pitahaya Red', 102: 'Apple Golden 1', 103: 'Pomelo Sweetie', 104: 'Cherry Rainier', 105: 'Avocado', 106: 'Apple Red Yellow 2', 107: 'Raspberry', 108: 'Mangostan', 109: 'Strawberry Wedge', 110: 'Kaki', 111: 'Mandarine', 112: 'Potato Sweet', 113: 'Cucumber Ripe 2', 114: 'Kumquats', 115: 'Pear Red', 116: 'Ginger Root', 117: 'Physalis', 118: 'Pear Kaiser', 119: 'Peach', 120: 'Corn', 121: 'Grape White 3', 122: 'Apple Red Yellow 1', 123: 'Grape Pink', 124: 'Banana', 125: 'Grape White 4', 126: 'Kohlrabi', 127: 'Pepper Green', 128: 'Watermelon', 129: 'Mango', 130: 'Tomato 2'}
        for i in range(5):
            res_labels.append(classes[int(top5label[0][i])])
            res_proba.append(math.exp(float(top5prob[0][i])))
        QMessageBox.about(self, "Predictions", 'Classes: ' + ' '.join(str(e) for e in res_labels) + '\nProbabilities: ' + ' '.join(str(round(e, 2)) for e in res_proba))
        return


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
