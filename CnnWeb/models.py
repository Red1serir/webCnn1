import os
from django.db import models
from django.dispatch import receiver
from django.utils.translation import ugettext_lazy as _

# Create your models here.
class Image(models.Model):
    image = models.ImageField(upload_to="images/")




@receiver(models.signals.post_delete, sender=Image)
def auto_delete_file_on_delete(sender, instance, **kwargs):
    """
    Deletes file from filesystem
    when corresponding MediaFile object is deleted.
    """
    if instance.image:
        if os.path.isfile(instance.image.path):
            os.remove(instance.image.path)

