
from django import forms

class Image(forms.Form):
    image = forms.ImageField()