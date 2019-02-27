from distutils.core import setup, Extension

taramodule = Extension ('tara',
                       sources = ['pyinitcamera.c'],
					include_dirs = [('/usr/lib/aarch64-linux-gnu/glib-2.0/include'),
					 				('/usr/include/glib-2.0')])
setup (name = 'tarapackage',
       version = '1.0',
       description = 'This is a Tara Extension Unit package',
       ext_modules = [taramodule])
