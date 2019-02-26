========================================================================
    SHARED OBJECT LIBRARY : xunit_lib_tara Project Overview
========================================================================

This library defines the functions for HID commands.
The following HID commands are implemented,

        (i)     InitExtensionUnit
        (ii)    DeinitExtensionUnit
        (iii)   ReadFirmwareVersion
        (iv)    GetCameraUniqueID
        (v)     GetManualExposureStereo
        (vi)    SetManualExposureStereo
        (vii)   SetAutoExposureStereo
        (viii)  GetIMUConfig
        (ix)    SetIMUConfig
        (x)     ControlIMUCapture
        (xi)    GetIMUValueBuffer
        (xii)   StereoCalibRead
        (xiii)  GetStreamModeStereo
        (xiv)   SetStreamModeStereo
        (xv)    SetHDRModeStereo
        (xvi)   GetHDRModeStereo
        (xvii)  GetIMUTemperatureData

Note:
	For detailed explanation, please refer to the Tara_Linux_Extension_Unit_API_Document.pdf
	document provided in the release package.

	It is not recommended to add or modify other than the implemented command formats. 
	
Command to create libecon_xunit.so:
===================================
To Build:
		$ make
		
To clean:
		$ make clean				
====================================
