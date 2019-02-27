from distutils.core import setup, Extension

taramodule = Extension ('tara',
                       sources = ['pyinitcamera.c'])
setup (name = 'tarapackage',
       version = '1.0',
       description = 'This is a Tara Extension Unit package',
       ext_modules = [taramodule])
