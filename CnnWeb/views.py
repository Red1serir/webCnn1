from django.shortcuts import render
from .apps import autoencoder
from .forms import Image as ImageForm
# Create your views here.
import numpy as np
from PIL import Image
from .models import Image as ImageDB
from skimage.transform import  resize
from  matplotlib  import  pyplot
def index(request):

    image = ImageForm(request.POST, request.FILES)
    if image.is_valid():

        ImageDB.objects.all().delete()
        imagedb = ImageDB()
        imagedb.image = image.cleaned_data['image']
        imagedb.save()

        data = []
        #img = Image.open(imagedb.image.path)
        #img.thumbnail((256, 256), Image.ANTIALIAS)
        img = pyplot.imread(imagedb.image.path)
        img=resize(img ,(256, 256))
        #img = resize(resize(img, (90, 90)), (256, 256))
        #img_array = np.array(img).astype(np.float32) / 255
        data.append(np.array(img))
        print('----------->',np.array(data).shape)
        gen_img_arry = np.clip(autoencoder.predict(np.array(data)), 0.0, 1.0)
        #gen_img_arry = np.clip(autoencoder.predict(gen_img_arry), 0.0, 1.0)
        gen_img = Image.fromarray(np.uint8(np.asarray(gen_img_arry*255))[0],'RGB')
        print("---------->>>",imagedb.image.name)
        path = 'images/generated_'+imagedb.image.name
        gen_img.save(path)
        return render(request, 'CnnWeb/index.html', {'url_image' : '/media/'+path.replace('/generated_images',''),'url_image_generated': '/media/'+path})

    return render(request, 'CnnWeb/index.html', { 'url_image' : None ,'url_image_generated': None})
