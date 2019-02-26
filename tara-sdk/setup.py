from distutils.core import setup, Extension

xunitmodule = Extension ('xunit',
						 sources = ['PyXunit.cpp'])
setup (name = 'xunitpackage',
       version = '1.0',
       description = 'This is a Tara Extension Unit package',
       ext_modules = [xunitmodule])
