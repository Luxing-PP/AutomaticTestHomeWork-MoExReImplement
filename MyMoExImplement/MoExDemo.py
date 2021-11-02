import torchvision
from torchvision.transforms import transforms
from PIL import Image


def normalize(x):
    mean = x.mean(dim=1, keepdim=True)
    std = x.var(dim=1, keepdim=True).sqrt()
    x = (x - mean) / std
    return x, mean, std


def momentExchange(x, beta, gamma):
    x.mul_(gamma)
    x.add_(beta)
    return x


def MoEx(featureImage, momentImage):
    transformImgToTen = transforms.Compose([
        transforms.ToTensor()
    ])

    transformTenToImg = transforms.Compose([
        transforms.ToPILImage()
    ])

    tensor1 = transformImgToTen(featureImage)
    tensor2 = transformImgToTen(momentImage)

    # MoEx
    hA, mA, vA = normalize(tensor1)
    hB, mB, vB = normalize(tensor2)
    MoExOutput = momentExchange(hA, mB, vB)
    return transformTenToImg(MoExOutput)


# implement by standard Normalize , show the basic concept/idea of MoEx
if __name__ == '__main__':
    # 大体上认为变换完的label应该还是与FeatureImage本来的label相同
    FeatureImage = Image.open("ImageSample/image (1).jpg")
    MomentImage = Image.open("ImageSample/image (2).jpg")

    PIL = MoEx(FeatureImage, MomentImage)
    PIL.show()
