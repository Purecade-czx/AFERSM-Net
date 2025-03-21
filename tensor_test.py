import torch
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
import numpy as np
from PIL import Image

# loader使用torchvision中自带的transforms函数
loader = transforms.Compose([
    transforms.ToTensor()])

unloader = transforms.ToPILImage()

a = torch.FloatTensor(3,3)
image_size = 6
val = [1,2,3,4,5,6]
gasf = GramianAngularField(image_size=image_size, method='difference')
data = np.array(val)
data = data.reshape(1,-1)
X_gasf = gasf.fit_transform(data)
X_gasf = torch.tensor(X_gasf)
X_gasf = X_gasf.squeeze()
print(X_gasf.shape)
plt.plot(X_gasf)
plt.show()
