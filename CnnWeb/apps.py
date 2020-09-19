from django.apps import AppConfig

from .model import model
autoencoder = model()
class CnnwebConfig(AppConfig):
    name = 'CnnWeb'
