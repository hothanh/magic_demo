/*-------------------------------------------------
* Python wrapper for C++ Extension Unit Functions
*------------------------------------------------*/

#include <Python.h>

///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2016, e-Con Systems.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS.
// IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR 
// ANY DIRECT/INDIRECT DAMAGES HOWEVER CAUSED AND ON ANY THEORY OF 
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
////////////////////////  ///////////////////////////////////////////////////

/**********************************************************************
	xunit_lib_tara.cpp : Defines the extension unit functions
						 in the shared library.
	Definitions for all HID commands.
**********************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <linux/input.h>
#include <linux/hidraw.h>

//#include "xunit_lib_tara.h"


/* For Stereo - Tara */
/* Commands */
#define CAMERA_CONTROL_STEREO			0x78

#define READFIRMWAREVERSION			0x40
#define GETCAMERA_UNIQUEID			0x41
	
#define GET_EXPOSURE_VALUE			0x01
#define SET_EXPOSURE_VALUE			0x02
#define SET_AUTO_EXPOSURE			0x02

#define GET_IMU_CONFIG				0x03
#define SET_IMU_CONFIG				0x04
#define CONTROL_IMU_VAL				0x05
#define SEND_IMU_VAL_BUFF			0x06
	
#define READ_CALIB_REQUEST			0x09
#define READ_CALIB_DATA				0x0A

#define SET_STREAM_MODE_STEREO			0x0B
#define GET_STREAM_MODE_STEREO			0x0C
#define GET_IMU_TEMP_DATA			0x0D
#define SET_HDR_MODE_STEREO			0x0E
#define GET_HDR_MODE_STEREO			0x0F
	
#define IMU_NUM_OF_VAL				0xFF
#define IMU_ACC_VAL				0xFE
#define IMU_GYRO_VAL				0xFD
#define INTRINSIC_FILEID			0x00
#define EXTRINSIC_FILEID			0x01
#define PCK_SIZE				56

#define TRUE                     		1
#define FALSE                    		0

#include <stdbool.h>
#include <libudev.h>
#include <pthread.h>

#define VID			"2560"
#define See3CAM_STEREO		"c114"

typedef bool 	  		 BOOL;
typedef int8_t    		 INT8;
typedef int16_t  		 INT16;
typedef int32_t   		 INT32;
typedef unsigned char		 UINT8;
typedef unsigned short int   	 UINT16;
typedef unsigned int		 UINT32;

/* Stereo IMU */
typedef struct {
	INT8 IMU_MODE;
	INT8 ACC_AXIS_CONFIG;
	INT8 ACC_SENSITIVITY_CONFIG;
	INT8 GYRO_AXIS_CONFIG;
	INT8 GYRO_SENSITIVITY_CONFIG;
	INT8 IMU_ODR_CONFIG;
} IMUCONFIG_TypeDef;

typedef struct {
	INT8 IMU_UPDATE_MODE;
	UINT16 IMU_NUM_OF_VALUES;
} IMUDATAINPUT_TypeDef;

typedef struct {
	UINT16 IMU_VALUE_ID;
	double accX;
	double accY;
	double accZ;
	double gyroX;
	double gyroY;
	double gyroZ;
} IMUDATAOUTPUT_TypeDef;

/* Report Numbers */
#define SET_FAIL				0x00
#define SET_SUCCESS				0x01
#define GET_FAIL				0x00
#define GET_SUCCESS				0x01
#define SUCCESS			 		1
#define FAILURE					-1

#define BUFFER_LENGTH				65
#define TIMEOUT					2000
#define CALIB_TIMEOUT				5000
#define DESCRIPTOR_SIZE_ENDPOINT		29
#define DESCRIPTOR_SIZE_IMU_ENDPOINT		23

/* HID STATUS */
#define SEE3CAM_STEREO_HID_SUCCESS		(0x01)
#define SEE3CAM_STEREO_HID_FAIL			(0x00)

/* EXPOSURE CONTROL */
#define SEE3CAM_STEREO_EXPOSURE_AUTO		(1)
#define SEE3CAM_STEREO_EXPOSURE_MIN		(10)
#define SEE3CAM_STEREO_EXPOSURE_MAX		(1000000)
#define SEE3CAM_STEREO_EXPOSURE_DEF		(8000)

/* IMU MODE */
#define IMU_ACC_GYRO_DISABLE			(0x00)
#define IMU_ACC_ENABLE				(0x01)
#define IMU_ACC_GYRO_ENABLE			(0x03)

/* ACC AXIS CONTROL */
#define IMU_ACC_X_Y_Z_ENABLE			(0x07)
#define IMU_ACC_X_ENABLE			(0x01)
#define IMU_ACC_Y_ENABLE			(0x02)
#define IMU_ACC_Z_ENABLE			(0x04)

/* ACC ODR CONTROL */
#define IMU_ODR_10_14_9HZ			(0x01)
#define IMU_ODR_50_59_9HZ			(0x02)
#define IMU_ODR_119HZ				(0x03)
#define IMU_ODR_238HZ				(0x04)
#define IMU_ODR_476HZ				(0x05)
#define IMU_ODR_952HZ				(0x06)

/* ACC SENSITIVITY CONTROL */
#define IMU_ACC_SENS_2G				(0x00)
#define IMU_ACC_SENS_4G				(0x02)
#define IMU_ACC_SENS_8G				(0x03)
#define IMU_ACC_SENS_16G			(0x01)

/* GYRO AXIS CONTROL */
#define IMU_GYRO_X_Y_Z_ENABLE			(0x07)
#define IMU_GYRO_X_ENABLE			(0x01)
#define IMU_GYRO_Y_ENABLE			(0x02)
#define IMU_GYRO_Z_ENABLE			(0x04)

/* GYRO SENSITIVITY CONTROL */
#define IMU_GYRO_SENS_245DPS			(0x00)
#define IMU_GYRO_SENS_500DPS			(0x01)
#define IMU_GYRO_SENS_2000DPS			(0x03)

/* IMU VALUE UPDATE MODE */
#define IMU_CONT_UPDT_EN			(0x01)
#define IMU_CONT_UPDT_DIS			(0x02)

/* IMU VALUES CONTROL */
#define IMU_AXES_VALUES_MIN			(1)
#define IMU_AXES_VALUES_MAX			(65535)

/* Range of Gyro */
#define LSM6DS0_G_FS_245                   	(UINT8)(0x00) /* Full scale: 245 dps  */
#define LSM6DS0_G_FS_500                    	(UINT8)(0x08) /* Full scale: 500 dps  */
#define LSM6DS0_G_FS_2000                   	(UINT8)(0x18) /* Full scale: 2000 dps */

/* Range of Accelero */
#define LSM6DS0_XL_FS_2G                    	(UINT8)(0x00) /* Full scale: +- 2g */
#define LSM6DS0_XL_FS_4G                    	(UINT8)(0x10) /* Full scale: +- 4g */
#define LSM6DS0_XL_FS_8G                    	(UINT8)(0x18) /* Full scale: +- 8g */
#define LSM6DS0_XL_FS_16G                   	(UINT8)(0x08) /* Full scale: +- 16g*/
//Global IMU variables.

/* Function Declarations */

BOOL InitExtensionUnit (char *busname);				//Initializes the Extension unit 

BOOL DeinitExtensionUnit (void);				//Deinits the Extension unit 

BOOL ReadFirmwareVersion (UINT8 *pMajorVersion, UINT8 *pMinorVersion1, UINT16 *pMinorVersion2, UINT16 *pMinorVersion3);
								//Reads the Firmware version of the device.

BOOL GetCameraUniqueID (char *UniqueID);			//Reads the unique ID of the camera

BOOL GetManualExposureStereo (INT32 *ManualExposureValue);	//Outputs the current Exposure Value of the camera

BOOL SetManualExposureStereo (INT32 ManualExposureValue);	//Sets the Exposure Value passed to the camera

BOOL SetAutoExposureStereo (void);				//Sets the Camera to Auto Exposure

BOOL GetIMUConfig (IMUCONFIG_TypeDef *lIMUConfig);		//Reads the current configuration of the IMU
	
BOOL SetIMUConfig (IMUCONFIG_TypeDef lIMUConfig);		//Sets the IMU configuration

BOOL ControlIMUCapture (IMUDATAINPUT_TypeDef *lIMUInput);	//Configures the IMU to read the IMU data in a specific format

BOOL GetIMUValueBuffer (pthread_mutex_t *lIMUDataReadyEvent, IMUDATAOUTPUT_TypeDef *lIMUAxes);	
								//Reads the IMU values

BOOL StereoCalibRead (unsigned char **IntrinsicBuffer, unsigned char **ExtrinsicBuffer, int *lIntFileLength, int *lExtFileLength);
								//Reads back the Intrinsic and Extrinsic values of the camera from the flash

BOOL GetStreamModeStereo (UINT32 *iStreamMode);			//Reads the mode in which the camera is set

BOOL SetStreamModeStereo (UINT32 iStreamMode);			//Sets the mode passed in the camera 

BOOL SetHDRModeStereo (UINT32 HDRMode);				//Enables/Disables the HDR mode

BOOL GetHDRModeStereo (UINT32 *HDRMode);			//Reads the status of the HDR mode

BOOL GetIMUTemperatureData (UINT8 *iMSBTemp, UINT8 *iLSBTemp);	//Reads the temperature data of the IMU unit


/* Function Declarations */

const char *bus_str (int);

unsigned int GetTickCount (void);

int find_hid_device (char *);

IMUCONFIG_TypeDef				glIMUConfig;
IMUDATAINPUT_TypeDef				glIMUInput;
			
BOOL						g_IsIMUConfigured = FALSE;
float						glAccSensMult = 0;
float						glGyroSensMult = 0;
			
int 						hid_fd = -1, hid_imu = -1;
int 						countHidDevices = 0;
			
unsigned char					g_out_packet_buf[BUFFER_LENGTH];
unsigned char 					g_in_packet_buf[BUFFER_LENGTH];
const char					*hid_device;
const char					*hid_device_array[2];


//Auxiliary Functions
void Sleep(unsigned int TimeInMilli)
{
	BOOL timeout = TRUE;
	unsigned int start, end = 0;

	start = GetTickCount();
	while(timeout)
	{
		end = GetTickCount();
		if(end - start > TimeInMilli)
		{
			timeout = FALSE;
		}
	}
	return;
}


/*
  **********************************************************************************************************
 *  MODULE TYPE	:	LIBRAY API 							    *
 *  Name	:	InitExtensionUnit						    *
 *  Parameter1	:	char* (busname)							    *
 *  Returns	:	BOOL (TRUE or FALSE)						    *
 *  Description	:	Finds hidraw device based on the busname and initialize the HID device 		*
			Application should call this function before calling any other function		*
  **********************************************************************************************************
*/
BOOL InitExtensionUnit(char *busname)
{
	int index, fd, ret, desc_size = 0;
	char buf[256];
	struct hidraw_devinfo info;
	struct hidraw_report_descriptor rpt_desc;
	countHidDevices = 0;
	ret = find_hid_device(busname);
	if(ret < 0)
	{
		printf("%s(): Not able to find the e-con's see3cam device\n", __func__);
		return FALSE;
	}
	
	printf("\nHERE\n");
	//printf("count HID devices : %d\n", countHidDevices);
	for(index=0; index < countHidDevices; index++)
	{
		//printf(" Selected HID Device : %s\n",hid_device_array[index]);

		/* Open the Device with non-blocking reads. In real life,
		   don't use a hard coded path; use libudev instead. */
		fd = open(hid_device_array[index], O_RDWR|O_NONBLOCK);

		if (fd < 0) {
			perror("xunit-InitExtensionUnit : Unable to open device");
			return FALSE;
		}

		memset(&rpt_desc, 0x0, sizeof(rpt_desc));
		memset(&info, 0x0, sizeof(info));
		memset(buf, 0x0, sizeof(buf));

		/* Get Report Descriptor Size */
		ret = ioctl(fd, HIDIOCGRDESCSIZE, &desc_size);
		if (ret < 0) {
			perror("xunit-InitExtensionUnit : HIDIOCGRDESCSIZE");
			return FALSE;
		}

		//printf("Report Descriptor Size: %d\n", desc_size);

		/* Get Report Descriptor */
		rpt_desc.size = desc_size;
		ret = ioctl(fd, HIDIOCGRDESC, &rpt_desc);
		if (ret < 0) {
			perror("xunit-InitExtensionUnit : HIDIOCGRDESC");
			return FALSE;
		}

		/*printf("Report Descriptors:\n");
		for (i = 0; i < rpt_desc.size; i++)
			printf("%hhx ", rpt_desc.value[i]);
		puts("\n");*/


		/* Get Raw Name */
		ret = ioctl(fd, HIDIOCGRAWNAME(256), buf);
		if (ret < 0) {
			perror("xunit-InitExtensionUnit : HIDIOCGRAWNAME");
			return FALSE;
		}
		
		//printf("Raw Name: %s\n", buf);

		/* Get Physical Location */
		ret = ioctl(fd, HIDIOCGRAWPHYS(256), buf);
		if (ret < 0) {
			perror("xunit-InitExtensionUnit : HIDIOCGRAWPHYS");
			return FALSE;
		}

		//printf("Raw Phys: %s\n", buf);

		/* Get Raw Info */
		ret = ioctl(fd, HIDIOCGRAWINFO, &info);
		if (ret < 0) {
			perror("xunit-InitExtensionUnit : HIDIOCGRAWINFO");
			return FALSE;
		}
		
		/*printf("Raw Info:\n");
		printf("\tbustype: %d (%s)\n", info.bustype, bus_str(info.bustype));
		printf("\tvendor: 0x%04hx\n", info.vendor);
		printf("\tproduct: 0x%04hx\n", info.product);*/


		if(desc_size == DESCRIPTOR_SIZE_ENDPOINT)
		{
			hid_fd = fd;
			//printf("hid_fd = %d\n", hid_fd);
		}
		else if(desc_size == DESCRIPTOR_SIZE_IMU_ENDPOINT)
		{
			hid_imu = fd;
			//printf("hid_imu = %d\n", hid_imu);
		}
	}

	return TRUE;
}


/*
  **********************************************************************************************************
 *  MODULE TYPE	:	LIBRAY API 						*
 *  Name	:	ReadFirmwareVersion					*
 *  Parameter1	:	unsigned char *		(Major Version)			*
 *  Parameter2	:	unsigned char *		(Minor Version 1)		*
 *  Parameter3	:	unsigned short int * 	(Minor Version 2)		*
 *  Parameter4	:	unsigned short int * 	(Minor Version 3)		*
 *  Returns	:	BOOL (TRUE or FALSE)					*
 *  Description	:   	Sends the extension unit command for reading firmware version to the UVC device	*
 *			and then device sends back the firmware version will be stored in the variables	*	
  **********************************************************************************************************
*/
BOOL ReadFirmwareVersion (UINT8 *pMajorVersion, UINT8 *pMinorVersion1, UINT16 *pMinorVersion2, UINT16 *pMinorVersion3)
{	

	BOOL timeout = TRUE;
	int ret = 0;
	unsigned int start, end = 0;
	unsigned short int sdk_ver=0, svn_ver=0;
	
	//Initialize the buffer
	memset(g_out_packet_buf, 0x00, sizeof(g_out_packet_buf));

	//Set the Report Number
	g_out_packet_buf[1] = READFIRMWAREVERSION; 	/* Report Number */

	/* Send a Report to the Device */
	ret = write(hid_fd, g_out_packet_buf, BUFFER_LENGTH);
	if (ret < 0) {
		perror("xunit-ReadFirmwareVersion : write failed");
		return FALSE;
	} else {
		//printf("%s(): wrote %d bytes\n", __func__,ret);
	}
	/* Read the Firmware Version from the device */
	start = GetTickCount();
	while(timeout) 
	{	
		/* Get a report from the device */
		ret = read(hid_fd, g_in_packet_buf, BUFFER_LENGTH);
		if (ret < 0) {
			//perror("read");
		} else {
			//printf("%s(): read %d bytes:\n", __func__,ret);
			if(g_in_packet_buf[0] == READFIRMWAREVERSION) {
				sdk_ver = (g_in_packet_buf[3]<<8)+g_in_packet_buf[4];
				svn_ver = (g_in_packet_buf[5]<<8)+g_in_packet_buf[6];

				*pMajorVersion = g_in_packet_buf[1];
				*pMinorVersion1 = g_in_packet_buf[2];
				*pMinorVersion2 = sdk_ver;
				*pMinorVersion3 = svn_ver;

				timeout = FALSE;
			}
	 	}
		end = GetTickCount();
		if(end - start > TIMEOUT)
		{
			printf("%s(): Timeout occurred\n", __func__);
			timeout = FALSE;
			return FALSE;
		}		
	}
	return TRUE;
}


/*
  **********************************************************************************************************
 *  MODULE TYPE	:	LIBRAY API 					*
 *  Name	:	GetCameraUniqueID				*
 *  Parameter1	:	char * (UniqueID)				*
 *  Returns	:	BOOL (TRUE or FALSE)				*
 *  Description	:	Sends the extension unit command for reading serial number to the UVC device	*
 *			and then device sends back the unique ID which will be stored in UniqueID	*	
  **********************************************************************************************************
*/
BOOL GetCameraUniqueID (char *UniqueID)
{	

	BOOL timeout = TRUE;
	int ret = 0;
	int i,k,tmp = 0;
	unsigned int start, end = 0;
	UniqueID[BUFFER_LENGTH] = '\0';
	
	//Initialize the buffer
	memset(g_out_packet_buf, 0x00, sizeof(g_out_packet_buf));
	memset(UniqueID, 0x00, sizeof(UniqueID[BUFFER_LENGTH]));

	//Set the Report Number
	g_out_packet_buf[1] = GETCAMERA_UNIQUEID; 	/* Report Number */

	/* Send a Report to the Device */
	ret = write(hid_fd, g_out_packet_buf, BUFFER_LENGTH);
	if (ret < 0) {
		perror("xunit-GetCameraUniqueID : write failed");
		return FALSE;
	} else {
		//printf("%s(): wrote %d bytes\n", __func__,ret);
	}
	/* Read the Serial Number from the device */
	start = GetTickCount();
	while(timeout) 
	{	
		/* Get a report from the device */
		ret = read(hid_fd, g_in_packet_buf, BUFFER_LENGTH);
		if (ret < 0) {
			//perror("read");
		} else {
			//printf("%s(): read %d bytes:\n", __func__,ret);
			if(g_in_packet_buf[0] == GETCAMERA_UNIQUEID) {
				for(i=1,k=3;i<5;i++,k--)
					tmp |= g_in_packet_buf[i]<<(k*8);
				sprintf(UniqueID,"%X",tmp);
				//printf("\n\nUnique ID is : %s\n", UniqueID);
				timeout = FALSE;
			}
	 	}
		end = GetTickCount();
		if(end - start > TIMEOUT)
		{
			printf("%s(): Timeout occurred\n", __func__);
			timeout = FALSE;
			return FALSE;
		}		
	}
	return TRUE;
}


/*
  **********************************************************************************************************
 *  MODULE TYPE	:	LIBRAY API 					*
 *  Name	:	DeinitExtensionUnit				*
 *  Returns	:	BOOL (TRUE or FALSE)				*
 *  Description	:  	To release all the extension unit objects and other internal library objects    *	
  **********************************************************************************************************
*/
BOOL DeinitExtensionUnit()
{
	int ret=0;
	/* Close the hid fd */
	if(hid_fd > 0)
	{
		ret=close(hid_fd);
	}
	if(ret<0)
		return FALSE ;
	else
		return TRUE;
	//When threads are used don't forget to terminate them here.
}


/*
  **************************************************************************************************************
 *  MODULE TYPE	:	LIBRAY API 						*
 *  Name	:	GetManualExposureStereo					*
 *  Parameter1	:	INT32* (ExposureValue)				 	*
 *  Returns	:	BOOL (TRUE or FALSE)					*
 *  Description	:   	Sends the extension unit command to get the manual exposure value from the camera   *
 *			and then device sends back the exposure value which will be stored in ExposureValue *
  **************************************************************************************************************
*/
BOOL GetManualExposureStereo(INT32 *ExposureValue)
{
	BOOL timeout = TRUE;
	int ret = 0;
	unsigned int start, end = 0;

	//Initialize the buffer
	memset(g_out_packet_buf, 0x00, sizeof(g_out_packet_buf));

	//Set the Report Number
	g_out_packet_buf[1] = CAMERA_CONTROL_STEREO; 	/* Report Number */
	g_out_packet_buf[2] = GET_EXPOSURE_VALUE; 	/* Report Number */

	/* Send a Report to the Device */
	ret = write(hid_fd, g_out_packet_buf, BUFFER_LENGTH);
	if (ret < 0) {
		perror("xunit-GetManualExposureValue_Stereo : write failed");
		return FALSE;
	} else {
		//printf("%s(): wrote %d bytes\n", __func__,ret);
	}

	/* Read the status from the device */
	start = GetTickCount();
	while(timeout)
	{
		/* Get a report from the device */
		ret = read(hid_fd, g_in_packet_buf, BUFFER_LENGTH);
		if (ret < 0) {
			//perror("read");
		} else {
			//printf("%s(): read %d bytes:\n", __func__,ret);
			if(g_in_packet_buf[0] == CAMERA_CONTROL_STEREO &&
				g_in_packet_buf[1] == GET_EXPOSURE_VALUE ) {
					if(g_in_packet_buf[10] == GET_SUCCESS) {
						*ExposureValue = (INT32)(((g_in_packet_buf[2] & 0xFF) << 24)
								+ ((g_in_packet_buf[3] & 0xFF) << 16)
								+ ((g_in_packet_buf[4] & 0xFF) << 8)
								+ (g_in_packet_buf[5] & 0xFF)
								);
						timeout = FALSE;
					} else if(g_in_packet_buf[10] == GET_FAIL) {
						return FALSE;
					}
			}
	 	}
		end = GetTickCount();
		if(end - start > TIMEOUT)
		{
			printf("%s(): Timeout occurred\n", __func__);
			timeout = FALSE;
			return FALSE;
		}
	}
	return TRUE;
}


/*
  **********************************************************************************************************
 *  MODULE TYPE	:	LIBRAY API 						*
 *  Name	:	SetManualExposureStereo					*
 *  Parameter1	:	INT32	(ExposureValue)				    	*
 *  Returns	:	BOOL (TRUE or FALSE)					*
 *  Description	:   	Sends the extension unit command to set the manual exposure value to the camera   *
 *			The exposure value ranges from 1 to 1000,000	  			  	  *
  **********************************************************************************************************
*/
BOOL SetManualExposureStereo(INT32 ExposureValue)
{
	BOOL timeout = TRUE;
	int ret = 0;
	unsigned int start, end = 0;

	if((ExposureValue > SEE3CAM_STEREO_EXPOSURE_MAX) || (ExposureValue < SEE3CAM_STEREO_EXPOSURE_MIN))
	{
		printf("Set Manual Exposure failed : Input out of bounds\n");
		return FALSE;
	}

	//Initialize the buffer
	memset(g_out_packet_buf, 0x00, sizeof(g_out_packet_buf));

	//Set the Report Number
	g_out_packet_buf[1] = CAMERA_CONTROL_STEREO; 	/* Report Number */
	g_out_packet_buf[2] = SET_EXPOSURE_VALUE; 	/* Report Number */

	g_out_packet_buf[3] = (UINT8)((ExposureValue >> 24) & 0xFF);
	g_out_packet_buf[4] = (UINT8)((ExposureValue >> 16) & 0xFF);
	g_out_packet_buf[5] = (UINT8)((ExposureValue >> 8) & 0xFF);
	g_out_packet_buf[6] = (UINT8)(ExposureValue & 0xFF);

	/* Send a Report to the Device */
	ret = write(hid_fd, g_out_packet_buf, BUFFER_LENGTH);
	if (ret < 0) {
		perror("xunit-SetManualExposureValue_Stereo : write failed");
		return FALSE;
	} else {
		//printf("%s(): wrote %d bytes\n", __func__,ret);
	}

	/* Read the status from the device */
	start = GetTickCount();
	while(timeout)
	{
		/* Get a report from the device */
		ret = read(hid_fd, g_in_packet_buf, BUFFER_LENGTH);
		if (ret < 0) {
			//perror("read");
		} else {
			//printf("%s(): read %d bytes:\n", __func__,ret);
			if(g_in_packet_buf[0] == CAMERA_CONTROL_STEREO &&
							g_in_packet_buf[1] == SET_EXPOSURE_VALUE){
					if(g_in_packet_buf[10] == SET_SUCCESS) {
						timeout = FALSE;
					} else if(g_in_packet_buf[10] == SET_FAIL) {
						return FALSE;
					}
			}
	 	}
		end = GetTickCount();
		if(end - start > TIMEOUT)
		{
			printf("%s(): Timeout occurred\n", __func__);
			timeout = FALSE;
			return FALSE;
		}
	}
	return TRUE;
}


/*
  **********************************************************************************************************
 *  MODULE TYPE	:	LIBRAY API 				*
 *  Name	:	SetAutoExposureStereo			*
 *  Returns	:	BOOL (TRUE or FALSE)			*
 *  Description	:   	Sends the extension unit command to set the camera to auto exposure.   *
  **********************************************************************************************************
*/
BOOL SetAutoExposureStereo()
{
	BOOL timeout = TRUE;
	int ret = 0;
	unsigned int start, end = 0;
	INT32 ExposureValue = 1;

	//Initialize the buffer
	memset(g_out_packet_buf, 0x00, sizeof(g_out_packet_buf));

	//Set the Report Number
	g_out_packet_buf[1] = CAMERA_CONTROL_STEREO; 	/* Report Number */
	g_out_packet_buf[2] = SET_AUTO_EXPOSURE; 	/* Report Number */

	g_out_packet_buf[3] = (UINT8)((ExposureValue >> 24) & 0xFF);
	g_out_packet_buf[4] = (UINT8)((ExposureValue >> 16) & 0xFF);
	g_out_packet_buf[5] = (UINT8)((ExposureValue >> 8) & 0xFF);
	g_out_packet_buf[6] = (UINT8)(ExposureValue & 0xFF);

	/* Send a Report to the Device */
	ret = write(hid_fd, g_out_packet_buf, BUFFER_LENGTH);
	if (ret < 0) {
		perror("xunit-SetAutoExposureStereo : write failed");
		return FALSE;
	} else {
		//printf("%s(): wrote %d bytes\n", __func__,ret);
	}

	/* Read the status from the device */
	start = GetTickCount();
	while(timeout)
	{
		/* Get a report from the device */
		ret = read(hid_fd, g_in_packet_buf, BUFFER_LENGTH);
		if (ret < 0) {
			//perror("read");
		} else {
			//printf("%s(): read %d bytes:\n", __func__,ret);
			if(g_in_packet_buf[0] == CAMERA_CONTROL_STEREO &&
							g_in_packet_buf[1] == SET_AUTO_EXPOSURE){
					if(g_in_packet_buf[10] == SET_SUCCESS) {
						timeout = FALSE;
					} else if(g_in_packet_buf[10] == SET_FAIL) {
						return FALSE;
					}
			}
	 	}
		end = GetTickCount();
		if(end - start > TIMEOUT)
		{
			printf("%s(): Timeout occurred\n", __func__);
			timeout = FALSE;
			return FALSE;
		}
	}
	return TRUE;
}


/*
  **********************************************************************************************************
 *  MODULE TYPE	:	LIBRAY API 				*
 *  Name	:	IMUSensitivityConfig			*
 *  Returns	:	void					*
 *  Description	:   	Sets the sensitivity to be multiplied in a global variable.  *
  **********************************************************************************************************
*/
void IMUSensitivityConfig(IMUCONFIG_TypeDef lIMUConfig)
{
	switch (glIMUConfig.ACC_SENSITIVITY_CONFIG * 0x08)
	{
		case LSM6DS0_XL_FS_2G:

			glAccSensMult = 0.061;
			break;

		case LSM6DS0_XL_FS_4G:

			glAccSensMult = 0.122;
			break;

		case LSM6DS0_XL_FS_8G:

			glAccSensMult = 0.244;
			break;

		case LSM6DS0_XL_FS_16G:

			glAccSensMult = 0.732;
			break;
	}

	switch (glIMUConfig.GYRO_SENSITIVITY_CONFIG * 0x08)
	{
		case LSM6DS0_G_FS_245:

			glGyroSensMult =  0.00875;
			break;

		case LSM6DS0_G_FS_500:

			glGyroSensMult = 0.00175;
			break;

		case LSM6DS0_G_FS_2000:

			glGyroSensMult = 0.07;
			break;
	}

	//printf("IMUSensitivityConfig: A = %f G = %f\r\n",glAccSensMult,glGyroSensMult);
}


/*
  **********************************************************************************************************
 *  MODULE TYPE	:	LIBRAY API 					*
 *  Name	:	GetIMUConfig					*
 *  Parameter1	:	IMUCONFIG_TypeDef 	(*lIMUConfig) 		*
 *  Returns	:	BOOL (TRUE or FALSE)				*
 *  Description	:   	Sends the extension unit command to Get the current IMU configuration.  *
   **********************************************************************************************************
*/
BOOL GetIMUConfig(IMUCONFIG_TypeDef *lIMUConfig)
{
	BOOL timeout = TRUE;
	int ret = 0;
	unsigned int start, end = 0;

	//Initialize the buffer
	memset(g_out_packet_buf, 0x00, sizeof(g_out_packet_buf));

	//Set the Report Number
	g_out_packet_buf[1] = CAMERA_CONTROL_STEREO; 	/* Report Number */
	g_out_packet_buf[2] = GET_IMU_CONFIG; 		/* Report Number */

	/* Send a Report to the Device */
	ret = write(hid_fd, g_out_packet_buf, BUFFER_LENGTH);
	if (ret < 0) {
		perror("xunit-GetIMUConfig : write failed");
		return FALSE;
	} else {
		//printf("%s(): wrote %d bytes\n", __func__,ret);
	}

	/* Read the status from the device */
	start = GetTickCount();
	while(timeout)
	{
		/* Get a report from the device */
		ret = read(hid_fd, g_in_packet_buf, BUFFER_LENGTH);
		if (ret < 0) {
			//perror("read");
		} else {
			//printf("%s(): read %d bytes:\n", __func__,ret);
			if(g_in_packet_buf[0] == CAMERA_CONTROL_STEREO &&
				g_in_packet_buf[1] == GET_IMU_CONFIG ) {
					if(g_in_packet_buf[25] == GET_SUCCESS) {
						lIMUConfig->IMU_MODE				= g_in_packet_buf[2];
						lIMUConfig->ACC_AXIS_CONFIG			= g_in_packet_buf[5];
						lIMUConfig->IMU_ODR_CONFIG			= g_in_packet_buf[6];
						lIMUConfig->ACC_SENSITIVITY_CONFIG	= g_in_packet_buf[7];
						lIMUConfig->GYRO_AXIS_CONFIG		= g_in_packet_buf[10];
						lIMUConfig->GYRO_SENSITIVITY_CONFIG	= g_in_packet_buf[12];

						glIMUConfig			= *lIMUConfig;
						IMUSensitivityConfig(glIMUConfig);
						g_IsIMUConfigured	= TRUE;

						timeout = FALSE;
					} else if(g_in_packet_buf[25] == GET_FAIL) {
						return FALSE;
					}
			}
		}
		end = GetTickCount();
		if(end - start > TIMEOUT)
		{
			printf("%s(): Timeout occurred\n", __func__);
			timeout = FALSE;
			return FALSE;
		}
	}
	return TRUE;
}


/*
  **********************************************************************************************************
 *  MODULE TYPE	:	LIBRAY API 					*
 *  Name	:	SetIMUConfig					*
 *  Parameter1	:	IMUCONFIG_TypeDef 	(lIMUConfig)		*
 *  Returns	:	BOOL (TRUE or FALSE)				*
 *  Description	:   	Sends the extension unit command to Set custom IMU configuration.  *
  **********************************************************************************************************
*/
BOOL SetIMUConfig(IMUCONFIG_TypeDef lIMUConfig)
{
	BOOL timeout = TRUE;
	int ret = 0;
	unsigned int start, end = 0;

	//Initialize the buffer
	memset(g_out_packet_buf,0x00,BUFFER_LENGTH);

	if(lIMUConfig.IMU_MODE == IMU_ACC_GYRO_DISABLE)
	{
		//Set the Report Number
		g_out_packet_buf[1] = CAMERA_CONTROL_STEREO;
		g_out_packet_buf[2] = SET_IMU_CONFIG;
		g_out_packet_buf[3] = lIMUConfig.IMU_MODE;
		g_out_packet_buf[6] = 0x00;
		g_out_packet_buf[7] = 0x00;
		g_out_packet_buf[8] = 0x00;

		g_out_packet_buf[11] = 0x00;
		g_out_packet_buf[12] = 0x00;
		g_out_packet_buf[13] = 0x00;

		goto SKIP_IMU_CONFIG_ACC_GYRO_DISABLE;
	}
	if((lIMUConfig.IMU_MODE != IMU_ACC_ENABLE) && (lIMUConfig.IMU_MODE != IMU_ACC_GYRO_ENABLE))
	{
		printf("SetIMUConfig: Invalid ACC-GYRO enable mode\r\n");
		return FALSE;
	}

	if((lIMUConfig.ACC_AXIS_CONFIG < IMU_ACC_X_ENABLE) || (lIMUConfig.ACC_AXIS_CONFIG > IMU_ACC_X_Y_Z_ENABLE))
	{
		printf("SetIMUConfig: Invalid ACC AXIS enable mode\r\n");
		return FALSE;
	}

	if((lIMUConfig.IMU_ODR_CONFIG < IMU_ODR_10_14_9HZ) || (lIMUConfig.IMU_ODR_CONFIG > IMU_ODR_952HZ))
	{
		printf("SetIMUConfig: Invalid ACC ODR config\r\n");
		return FALSE;
	}

	if((lIMUConfig.ACC_SENSITIVITY_CONFIG < IMU_ACC_SENS_2G) || (lIMUConfig.ACC_SENSITIVITY_CONFIG > IMU_ACC_SENS_8G))
	{
		printf("SetIMUConfig: Invalid ACC SENSITIVITY config\r\n");
		return FALSE;
	}

	if(lIMUConfig.IMU_MODE == IMU_ACC_GYRO_ENABLE)
	{
		if((lIMUConfig.GYRO_AXIS_CONFIG < IMU_GYRO_X_ENABLE) || (lIMUConfig.GYRO_AXIS_CONFIG > IMU_GYRO_X_Y_Z_ENABLE))
		{
			printf("SetIMUConfig: Invalid GYRO AXIS enable mode\r\n");
			return FALSE;
		}

		if((lIMUConfig.GYRO_SENSITIVITY_CONFIG != IMU_GYRO_SENS_245DPS) && (lIMUConfig.GYRO_SENSITIVITY_CONFIG != IMU_GYRO_SENS_500DPS)
			&& (lIMUConfig.GYRO_SENSITIVITY_CONFIG != IMU_GYRO_SENS_2000DPS))
		{
			printf("SetIMUConfig: Invalid GYRO SENSITIVITY config\r\n");
			return FALSE;
		}
	}

	//Set the Report Number
	g_out_packet_buf[1] = CAMERA_CONTROL_STEREO;
	g_out_packet_buf[2] = SET_IMU_CONFIG;
	g_out_packet_buf[3] = lIMUConfig.IMU_MODE;
	g_out_packet_buf[6] = lIMUConfig.ACC_AXIS_CONFIG;
	g_out_packet_buf[7] = lIMUConfig.IMU_ODR_CONFIG;
	g_out_packet_buf[8] = lIMUConfig.ACC_SENSITIVITY_CONFIG;

	g_out_packet_buf[11] = lIMUConfig.GYRO_AXIS_CONFIG;
	g_out_packet_buf[12] = 0x00;
	g_out_packet_buf[13] = lIMUConfig.GYRO_SENSITIVITY_CONFIG;

SKIP_IMU_CONFIG_ACC_GYRO_DISABLE:
	/* Send a Report to the Device */
	ret = write(hid_fd, g_out_packet_buf, BUFFER_LENGTH);
	if (ret < 0) {
		perror("xunit-SetIMUConfig : write failed");
		return FALSE;
	} else {
		//printf("%s(): wrote %d bytes\n", __func__,ret);
	}

	/* Read the status from the device */
	start = GetTickCount();
	while(timeout)
	{
		/* Get a report from the device */
		ret = read(hid_fd, g_in_packet_buf, BUFFER_LENGTH);
		if (ret < 0) {
			//perror("read");
		} else {
			//printf("%s(): read %d bytes:\n", __func__,ret);
			if(g_in_packet_buf[0] == CAMERA_CONTROL_STEREO &&
				g_in_packet_buf[1] == SET_IMU_CONFIG ) {
					if(g_in_packet_buf[25] == SET_SUCCESS) {
						glIMUConfig			= lIMUConfig;
						IMUSensitivityConfig(glIMUConfig);
						g_IsIMUConfigured	= TRUE;
						timeout = FALSE;
					} else if(g_in_packet_buf[25] == SET_FAIL) {
						return FALSE;
					}
			}
	 	}
		end = GetTickCount();
		if(end - start > TIMEOUT)
		{
			printf("%s(): Timeout occurred\n", __func__);
			timeout = FALSE;
			return FALSE;
		}
	}
	return TRUE;
}


/*
  **********************************************************************************************************
 *  MODULE TYPE	:	LIBRAY API 					*
 *  Name	:	ControlIMUCapture				*
 *  Parameter1	:	IMUDATAINPUT_TypeDef 	(lIMUInput)		*   
 *  Returns	:	BOOL (TRUE or FALSE)				*
 *  Description	:   	Sends the extension unit command to control the output of the IMU.  *
  **********************************************************************************************************
*/
BOOL ControlIMUCapture(IMUDATAINPUT_TypeDef *lIMUInput)
{
	BOOL timeout = TRUE;
	int ret = 0;
	unsigned int start, end = 0;
	IMUCONFIG_TypeDef lIMUConfig;

	if(glIMUConfig.IMU_MODE == IMU_ACC_GYRO_DISABLE)
	{
		printf("ControlIMUCapture: IMU Disabled, Enable using SetIMUConfig\r\n");
		return FALSE;
	}

	if((lIMUInput->IMU_UPDATE_MODE != IMU_CONT_UPDT_EN) &&
		(lIMUInput->IMU_UPDATE_MODE != IMU_CONT_UPDT_DIS))
	{
		printf("ControlIMUCapture: Write File Failed\r\n");
		return FALSE;
	}


	//Initialize the buffer
	memset(g_out_packet_buf,0x00,BUFFER_LENGTH);

	//Set the Report Number
	g_out_packet_buf[1] = CAMERA_CONTROL_STEREO;
	g_out_packet_buf[2] = CONTROL_IMU_VAL;
	g_out_packet_buf[3] = lIMUInput->IMU_UPDATE_MODE;
	g_out_packet_buf[6] = IMU_NUM_OF_VAL;
	g_out_packet_buf[7] = 0x00;//(INT8)((lIMUInput.IMU_NUM_OF_VALUES & 0xFF00) >> 8);
	g_out_packet_buf[8] = 0x00;//(INT8)(lIMUInput.IMU_NUM_OF_VALUES & 0xFF);

	/* Send a Report to the Device */
	ret = write(hid_fd, g_out_packet_buf, BUFFER_LENGTH);
	if (ret < 0) {
		perror("xunit-ControlIMUCapture : write failed");
		return FALSE;
	} else {
		//printf("%s(): wrote %d bytes\n", __func__,ret);
	}

	/* Read the status from the device */
	start = GetTickCount();
	while(timeout)
	{
		/* Get a report from the device */
		ret = read(hid_fd, g_in_packet_buf, BUFFER_LENGTH);
		if (ret < 0) {
			//perror("read");
		} else {
			//printf("%s(): read %d bytes:\n", __func__,ret);
			if(g_in_packet_buf[0] == CAMERA_CONTROL_STEREO &&
				g_in_packet_buf[1] == CONTROL_IMU_VAL) {
					if(g_in_packet_buf[19] == SET_SUCCESS) {						
						glIMUInput.IMU_UPDATE_MODE		= lIMUInput->IMU_UPDATE_MODE;
						glIMUInput.IMU_NUM_OF_VALUES	= 0;
						if(g_IsIMUConfigured == FALSE) {
							if(!GetIMUConfig(&lIMUConfig)) {
								printf("ControlIMUCapture: GetIMUConfig Failed\n");
								return FALSE;
							}
							Sleep(10);
						}
						timeout = FALSE;
					} else if(g_in_packet_buf[19] == SET_FAIL) {
						glIMUInput.IMU_UPDATE_MODE = lIMUInput->IMU_UPDATE_MODE = IMU_CONT_UPDT_DIS;	
						glIMUInput.IMU_NUM_OF_VALUES = lIMUInput->IMU_NUM_OF_VALUES = IMU_AXES_VALUES_MIN;
						return FALSE;
					}
			}
		}
		end = GetTickCount();
		if(end - start > TIMEOUT)
		{
			printf("%s(): Timeout occurred\n", __func__);
			timeout = FALSE;
			return FALSE;
		}
	}
	return TRUE;
}


/*
  **********************************************************************************************************
 *  MODULE TYPE	:	LIBRAY API 					*
 *  Name	:	GetIMUValueBuffer				*
 *  Parameter1	:	pthread_mutex_t (*IMUDataReadyEvent)		*
 *  Parameter2	:	IMUDATAOUTPUT_TypeDef (*lIMUAxes)		*
 *  Returns	:	BOOL (TRUE or FALSE)				*
 *  Description	:   	Sends the extension unit command to get the axis values from the IMU.  *
  **********************************************************************************************************
*/
BOOL GetIMUValueBuffer(pthread_mutex_t *IMUDataReadyEvent, IMUDATAOUTPUT_TypeDef *lIMUAxes)
{
	BOOL timeout = TRUE;
	int ret = 0;
	unsigned int start, end = 0;

	UINT16 lIDofValues = 0;
	IMUDATAOUTPUT_TypeDef *lIMUAxesInitAdd = lIMUAxes;

	if(glIMUConfig.IMU_MODE == IMU_ACC_GYRO_DISABLE)
	{
		printf("GetIMUValueBuffer: IMU Disabled, Enable using SetIMUConfig\r\n");
		return FALSE;
	}

	//Initialize the buffer
	memset(g_out_packet_buf,0x00,BUFFER_LENGTH);
	g_out_packet_buf[1] = CAMERA_CONTROL_STEREO;
	g_out_packet_buf[2] = SEND_IMU_VAL_BUFF;

	/* Send a Report to the Device */
	ret = write(hid_fd, g_out_packet_buf, BUFFER_LENGTH);
	if (ret < 0) {
		perror("xunit-GetIMUValueBuffer : write failed");
		return FALSE;
	} else {
		//printf("%s(): wrote %d bytes\n", __func__,ret);
	}

	for(lIDofValues = 0;((glIMUInput.IMU_UPDATE_MODE != IMU_CONT_UPDT_DIS) || (glIMUInput.IMU_NUM_OF_VALUES >= IMU_AXES_VALUES_MIN));)
	{
		/* Read the status from the device */
		timeout = TRUE;
		start = GetTickCount();
		while(timeout)
		{
			/* Get a report from the device */
			//memset(g_in_packet_buf,0x00,BUFFER_LENGTH);
			ret = read(hid_imu, g_in_packet_buf, BUFFER_LENGTH);
			if (ret < 0) {
				//printf("Error\n");
				//perror("read");
			} else {
				//printf("%s(): read %d bytes:\n", __func__,ret);
				if(g_in_packet_buf[0] == CAMERA_CONTROL_STEREO &&
					g_in_packet_buf[1] == SEND_IMU_VAL_BUFF) {
						if(g_in_packet_buf[48] == SET_SUCCESS) {

							lIMUAxes->IMU_VALUE_ID = ++lIDofValues;

							if(g_in_packet_buf[4] == IMU_ACC_VAL)
							{
								lIMUAxes->accX = (((INT16)((g_in_packet_buf[6]) | (g_in_packet_buf[5]<<8))) * glAccSensMult);
								lIMUAxes->accY = (((INT16)((g_in_packet_buf[8]) | (g_in_packet_buf[7]<<8))) * glAccSensMult);
								lIMUAxes->accZ = (((INT16)((g_in_packet_buf[10]) | (g_in_packet_buf[9]<<8))) * glAccSensMult);			
							}

							if(g_in_packet_buf[15] == IMU_GYRO_VAL)
							{
								lIMUAxes->gyroX = (((INT16)((g_in_packet_buf[17]) | (g_in_packet_buf[16]<<8))) * glGyroSensMult);
								lIMUAxes->gyroY = (((INT16)((g_in_packet_buf[19]) | (g_in_packet_buf[18]<<8))) * glGyroSensMult);
								lIMUAxes->gyroZ = (((INT16)((g_in_packet_buf[21]) | (g_in_packet_buf[20]<<8))) * glGyroSensMult);
							}

							if(glIMUInput.IMU_UPDATE_MODE == IMU_CONT_UPDT_EN)
							{
								if(lIMUAxes->IMU_VALUE_ID == IMU_AXES_VALUES_MAX)
								{
									lIMUAxes = lIMUAxesInitAdd;
									lIDofValues = 0;
								}
								else
									lIMUAxes++;
							}
							else
							{
								glIMUInput.IMU_NUM_OF_VALUES--;
								lIMUAxes++;
							}

							//Setting the event to tell the application the buffer is full.
							pthread_mutex_unlock(IMUDataReadyEvent);
							timeout = FALSE;
						} else if(g_in_packet_buf[48] == SET_FAIL) {
							return FALSE;
						}
				}
			}
			end = GetTickCount();
			if((end - start) > (2500))
			{
				printf("%s(): Timeout occurred\n", __func__);
				timeout = FALSE;
				return FALSE;
			}
		}
	}

	lIMUAxes--;
	glIMUInput.IMU_NUM_OF_VALUES = lIMUAxes->IMU_VALUE_ID ;
	lIMUAxes++;

	return TRUE;
}


/*
  **********************************************************************************************************
 *  MODULE TYPE	:	LIBRAY API 					*
 *  Name	:	StereoCalibRead					*
 *  Parameter1	:	unsigned char (**in_buffer)			*
 *  Parameter2	:	unsigned char (**ex_buffer)			*
 *  Parameter3	:	int *intFileLength				*
 *  Parameter4	:	int *extFileLength				*
 *  Returns	:	BOOL (TRUE or FALSE)				*
 *  Description	:  	Sends the extension unit command to read the calibration files stored in the flash. *
  **********************************************************************************************************
*/
BOOL StereoCalibRead(unsigned char **in_buffer, unsigned char **ex_buffer, int *intFileLength, int *extFileLength)
{
	BOOL timeout = TRUE;
	int ret = 0;
	unsigned int start, end = 0;

	int lIntFileLength = 0,lExtFileLength = 0;
	int lIntPckCnt = 0,lExtPckCnt = 0;
	int lLoopCount = 1;
	
	
	//1. Issue a Read request - Intrinsic file
	memset(g_out_packet_buf,0x00,BUFFER_LENGTH);

	g_out_packet_buf[1] = CAMERA_CONTROL_STEREO;
	g_out_packet_buf[2] = READ_CALIB_REQUEST;
	g_out_packet_buf[3] = INTRINSIC_FILEID;

	/* Send a Report to the Device */
	ret = write(hid_fd, g_out_packet_buf, BUFFER_LENGTH);
	if (ret < 0) {
		perror("xunit-StereoCalibRead : write failed");
		return FALSE;
	} else {
		//printf("%s(): wrote %d bytes\n", __func__,ret);
	}

	/* Read the status from the device */
	start = GetTickCount();
	while(timeout)
	{
		/* Get a report from the device */
		ret = read(hid_fd, g_in_packet_buf, BUFFER_LENGTH);
		if (ret < 0) {
			//perror("read");
		} else {
			//printf("%s(): read %d bytes:\n", __func__,ret);
			if(g_in_packet_buf[0] == CAMERA_CONTROL_STEREO &&
				g_in_packet_buf[1] == READ_CALIB_REQUEST) {
					if(g_in_packet_buf[15] == SEE3CAM_STEREO_HID_SUCCESS) {

						lIntFileLength = (UINT32)(((g_in_packet_buf[7] << 8 ) & 0xFF00) | (g_in_packet_buf[8] & 0xFF));

						lIntPckCnt = lIntFileLength / PCK_SIZE;
						if(lIntFileLength % PCK_SIZE != 0)
							lIntPckCnt++;
						timeout = FALSE;
					} else if(g_in_packet_buf[15] == SEE3CAM_STEREO_HID_FAIL) {
						printf("StereoCalibRead: Return Status Failed 1\r\n");
						return FALSE;
					}
			}
		}
		end = GetTickCount();
		if(end - start > CALIB_TIMEOUT)
		{
			printf("%s(): Timeout occurred\n", __func__);
			timeout = FALSE;
			return FALSE;
		}
	}
	Sleep(10);
	//Allocating the in buffer
	*in_buffer = (unsigned char*)calloc(lIntPckCnt * PCK_SIZE, sizeof(unsigned char));
	*intFileLength = lIntFileLength;
	
	if(*in_buffer == NULL)	
	{
		printf("Memory Allocation failed Intrinsic file\n");
		return FALSE;
	}
	

	//2.Issue a read data - Intrinsic file
	Sleep(10);
	
	for(lLoopCount = 1; lLoopCount < lIntPckCnt; )
	{
		memset(g_out_packet_buf,0x00,BUFFER_LENGTH);
		g_out_packet_buf[1] = CAMERA_CONTROL_STEREO;
		g_out_packet_buf[2] = READ_CALIB_DATA;
		g_out_packet_buf[3] = INTRINSIC_FILEID;

		/* Send a Report to the Device */
		ret = write(hid_fd, g_out_packet_buf, BUFFER_LENGTH);
		if (ret < 0) {
			perror("write");
			return FALSE;
		} else {
			//printf("%s(): wrote %d bytes\n", __func__,ret);
		}

		/* Read the status from the device */
		timeout = TRUE;
		start = GetTickCount();
		while(timeout)
		{
			/* Get a report from the device */
			ret = read(hid_fd, g_in_packet_buf, BUFFER_LENGTH);
			if (ret < 0) {
				//perror("read");
			} else {
				//printf("%s(): read %d bytes:\n", __func__,ret);
				if(g_in_packet_buf[0] == CAMERA_CONTROL_STEREO &&
					g_in_packet_buf[1] == READ_CALIB_DATA) {
						if(g_in_packet_buf[7] == SEE3CAM_STEREO_HID_SUCCESS) {

							lLoopCount = (UINT32)(((g_in_packet_buf[5] << 8 ) & 0xFF00) | (g_in_packet_buf[6] & 0xFF));		
							
							if(lLoopCount == lIntPckCnt)
							{
								memcpy(*in_buffer + ((lLoopCount - 1)*PCK_SIZE),&g_in_packet_buf[8],(lIntFileLength % PCK_SIZE));
								//printf("StereoCalibRead: Write File Passed 2\r\n");
							}
							else
							{
								memcpy(*in_buffer + ((lLoopCount - 1)*PCK_SIZE),&g_in_packet_buf[8],PCK_SIZE);
							}
							timeout = FALSE;
						} else if(g_in_packet_buf[7] == SEE3CAM_STEREO_HID_FAIL) {
							return FALSE;
						}
				}
			}
			end = GetTickCount();
			if(end - start > CALIB_TIMEOUT)
			{
				printf("%s(): Timeout occurred\n", __func__);
				timeout = FALSE;
				return FALSE;
			}
		}
		Sleep(10);
	}
	Sleep(100);
	
	
	//3. Issue a Read request - Extrinsic file
	timeout = TRUE;
	memset(g_out_packet_buf,0x00,BUFFER_LENGTH);
	g_out_packet_buf[1] = CAMERA_CONTROL_STEREO;
	g_out_packet_buf[2] = READ_CALIB_REQUEST;
	g_out_packet_buf[3] = EXTRINSIC_FILEID;

	/* Send a Report to the Device */
	ret = write(hid_fd, g_out_packet_buf, BUFFER_LENGTH);
	if (ret < 0) {
		perror("xunit-StereoCalibRead : write failed");
		return FALSE;
	} else {
		//printf("%s(): wrote %d bytes\n", __func__,ret);
	}

	/* Read the status from the device */
	start = GetTickCount();
	while(timeout)
	{
		/* Get a report from the device */
		ret = read(hid_fd, g_in_packet_buf, BUFFER_LENGTH);
		if (ret < 0) {
			//perror("read");
		} else {
			//printf("%s(): read %d bytes:\n", __func__,ret);
			if(g_in_packet_buf[0] == CAMERA_CONTROL_STEREO &&
				g_in_packet_buf[1] == READ_CALIB_REQUEST) {
					if(g_in_packet_buf[15] == SEE3CAM_STEREO_HID_SUCCESS) {

						lExtFileLength = (UINT32)(((g_in_packet_buf[7] << 8 ) & 0xFF00) | (g_in_packet_buf[8] & 0xFF));

						lExtPckCnt = lExtFileLength / PCK_SIZE;
						if(lExtFileLength % PCK_SIZE != 0)
							lExtPckCnt++;
						timeout = FALSE;
					} else if(g_in_packet_buf[15] == SEE3CAM_STEREO_HID_FAIL) {
						printf("StereoCalibRead: Return Status Failed 1\r\n");
						return FALSE;
					}
			}
		}
		end = GetTickCount();
		if(end - start > CALIB_TIMEOUT)
		{
			printf("%s(): Timeout occurred\n", __func__);
			timeout = FALSE;
			return FALSE;
		}
	}
	Sleep(10);
	
	//Allocating the ex buffer
	*ex_buffer = (unsigned char*)calloc(lExtPckCnt * PCK_SIZE, sizeof(unsigned char));
	*extFileLength = lExtFileLength;
	
	if(*ex_buffer == NULL)	
	{
		printf("Memory Allocation failed Intrinsic file\n");
		return FALSE;
	}

	
	//4.Issue a read data - Extrinsic file
	Sleep(10);

	for(lLoopCount = 1; lLoopCount < lExtPckCnt; )
	{
		memset(g_out_packet_buf,0x00,BUFFER_LENGTH);
		g_out_packet_buf[1] = CAMERA_CONTROL_STEREO;
		g_out_packet_buf[2] = READ_CALIB_DATA;
		g_out_packet_buf[3] = EXTRINSIC_FILEID;

		/* Send a Report to the Device */
		ret = write(hid_fd, g_out_packet_buf, BUFFER_LENGTH);
		if (ret < 0) {
			perror("write");
			return FALSE;
		} else {
			//printf("%s(): wrote %d bytes\n", __func__,ret);
		}

		/* Read the status from the device */
		timeout = TRUE;
		start = GetTickCount();
		while(timeout)
		{
			/* Get a report from the device */
			ret = read(hid_fd, g_in_packet_buf, BUFFER_LENGTH);
			if (ret < 0) {
				//perror("read");
			} else {
				//printf("%s(): read %d bytes:\n", __func__,ret);
				if(g_in_packet_buf[0] == CAMERA_CONTROL_STEREO &&
					g_in_packet_buf[1] == READ_CALIB_DATA) {
						if(g_in_packet_buf[7] == SEE3CAM_STEREO_HID_SUCCESS) {

							lLoopCount = (UINT32)(((g_in_packet_buf[5] << 8 ) & 0xFF00) | (g_in_packet_buf[6] & 0xFF));		
							
							if(lLoopCount == lExtPckCnt)
							{
								memcpy(*ex_buffer + ((lLoopCount - 1)*PCK_SIZE),&g_in_packet_buf[8],(lExtFileLength % PCK_SIZE));
								//printf("StereoCalibRead: Write File Passed 2\r\n");
							}
							else
							{
								memcpy(*ex_buffer + ((lLoopCount - 1)*PCK_SIZE),&g_in_packet_buf[8],PCK_SIZE);
							}
							timeout = FALSE;
						} else if(g_in_packet_buf[7] == SEE3CAM_STEREO_HID_FAIL) {
							return FALSE;
						}
				}
			}
			end = GetTickCount();
			if(end - start > CALIB_TIMEOUT)
			{
				printf("%s(): Timeout occurred\n", __func__);
				timeout = FALSE;
				return FALSE;
			}
		}
		Sleep(10);
	}
	Sleep(100);
	return TRUE;
}


/*
 *   **********************************************************************************************************
 *  MODULE TYPE	:	LIBRAY API 				*
 *  Name	:	GetStreamModeStereo			*
 *  Parameter1	:	UINT32* (iStreamMode)			*
 *  Returns	:	BOOL (TRUE or FALSE)			*
 *  Description	:  	Sends the extension unit command to read the mode in which the camera is set.	*
  **********************************************************************************************************
*/
BOOL GetStreamModeStereo(UINT32 *iStreamMode)
{
	BOOL timeout = TRUE;
	int ret = 0;
	unsigned int start, end = 0;

	//Initialize the buffer
	memset(g_out_packet_buf, 0x00, sizeof(g_out_packet_buf));

	//Set the Report Number
	g_out_packet_buf[1] = CAMERA_CONTROL_STEREO; 	/* Report Number */
	g_out_packet_buf[2] = GET_STREAM_MODE_STEREO; 		/* Report Number */

	/* Send a Report to the Device */
	ret = write(hid_fd, g_out_packet_buf, BUFFER_LENGTH);
	if (ret < 0) {
		perror("xunit-GetStreamModeStereo : write failed");
		return FALSE;
	} else {
		//printf("%s(): wrote %d bytes\n", __func__,ret);
	}

	/* Read the status from the device */
	start = GetTickCount();
	while(timeout)
	{
		/* Get a report from the device */
		ret = read(hid_fd, g_in_packet_buf, BUFFER_LENGTH);
		if (ret < 0) {
			//perror("read");
		} else {
			//printf("%s(): read %d bytes:\n", __func__,ret);
			if(g_in_packet_buf[0] == CAMERA_CONTROL_STEREO &&
				g_in_packet_buf[1] == GET_STREAM_MODE_STEREO ) {
					if(g_in_packet_buf[4] == GET_SUCCESS) {
						*iStreamMode = g_in_packet_buf[2];
						timeout = FALSE;
					} else if(g_in_packet_buf[4] == GET_FAIL) {
						return FALSE;
					}
			}
	 	}
		end = GetTickCount();
		if(end - start > TIMEOUT)
		{
			printf("%s(): Timeout occurred\n", __func__);
			timeout = FALSE;
			return FALSE;
		}
	}
	return TRUE;
}


/*
  **********************************************************************************************************
 *  MODULE TYPE	:	LIBRAY API 				*
 *  Name	:	SetStreamModeStereo			*
 *  Parameter1	:	UINT32 (iStreamMode)			*
 *  Returns	:	BOOL (TRUE or FALSE)			*
 *  Description	:   	Sends the extension unit command to set a particular stream mode.	*
  **********************************************************************************************************
*/
BOOL SetStreamModeStereo(UINT32 iStreamMode)
{
	BOOL timeout = TRUE;
	int ret = 0;
	unsigned int start, end = 0;

	//Initialize the buffer
	memset(g_out_packet_buf, 0x00, sizeof(g_out_packet_buf));

	//Set the Report Number
	g_out_packet_buf[1] = CAMERA_CONTROL_STEREO; 		/* Report Number */
	g_out_packet_buf[2] = SET_STREAM_MODE_STEREO; 		/* Report Number */
	g_out_packet_buf[3] = iStreamMode; 					/* Report Number */
	
	/* Send a Report to the Device */
	ret = write(hid_fd, g_out_packet_buf, BUFFER_LENGTH);
	if (ret < 0) {
		perror("xunit-SetStreamModeStereo : write failed");
		return FALSE;
	} else {
		//printf("%s(): wrote %d bytes\n", __func__,ret);
	}

	/* Read the status from the device */
	start = GetTickCount();
	while(timeout)
	{
		/* Get a report from the device */
		ret = read(hid_fd, g_in_packet_buf, BUFFER_LENGTH);
		if (ret < 0) {
			//perror("read");
		} else {
			//printf("%s(): read %d bytes:\n", __func__,ret);
			if(g_in_packet_buf[0] == CAMERA_CONTROL_STEREO &&
							g_in_packet_buf[1] == SET_STREAM_MODE_STEREO){
					if(g_in_packet_buf[4] == SET_SUCCESS) {
						timeout = FALSE;
					} else if(g_in_packet_buf[4] == SET_FAIL) {
						return FALSE;
					}
			}
	 	}
		end = GetTickCount();
		if(end - start > TIMEOUT)
		{
			printf("%s(): Timeout occurred\n", __func__);
			timeout = FALSE;
			return FALSE;
		}
	}
	return TRUE;
}


/*
  **********************************************************************************************************
 *  MODULE TYPE	:	LIBRAY API 			*
 *  Name	:	SetHDRModeStereo		*
 *  Parameter1	:	UINT32 (HDRMode)		*
 *  Returns	:	BOOL (TRUE or FALSE)		*
 *  Description	:   	Sends the extension unit command to set a particular HDR mode. *
  **********************************************************************************************************
*/
BOOL SetHDRModeStereo(UINT32 HDRMode)
{
	BOOL timeout = TRUE;
	int ret = 0;
	unsigned int start, end = 0;

	//Initialize the buffer	
	memset(g_out_packet_buf, 0x00, sizeof(g_out_packet_buf));
	
	//Set the Report Number
	g_out_packet_buf[1] = CAMERA_CONTROL_STEREO; 	/* Report Number */
	g_out_packet_buf[2] = SET_HDR_MODE_STEREO;
	g_out_packet_buf[3] = HDRMode;
	
	ret = write(hid_fd, g_out_packet_buf, BUFFER_LENGTH);
	if (ret < 0)
	{
		perror("xunit-GetHDRMode : write failed");
		return FALSE;
	}
	else
	{
		//printf("%s(): wrote %d bytes\n", __func__,ret);
	}
	
	start = GetTickCount();
	while (timeout)
	{
		/* Get a report from the device */
		ret = read(hid_fd, g_in_packet_buf, BUFFER_LENGTH);
		
		if (ret < 0)
		{
			//perror("read");
		}
		else
		{
			//printf("%s(): read %d bytes:\n", __func__,ret);
			if(g_in_packet_buf[0] == CAMERA_CONTROL_STEREO && g_in_packet_buf[1] == SET_HDR_MODE_STEREO)
			{
				if (  g_in_packet_buf[4] == SET_SUCCESS )
				{
					timeout = FALSE;
				}
				else
				{
					if ( g_in_packet_buf[4] == SET_FAIL )
					{
						return FALSE;
					}
				}
			}
	 	}
	 	end = GetTickCount();
		if(end - start > TIMEOUT)
		{
			printf("%s(): Timeout occurred\n", __func__);
			timeout = FALSE;
			return FALSE;
		}
	}
	return TRUE;
}


/*
  **********************************************************************************************************
 *  MODULE TYPE	:	LIBRAY API 				*
 *  Name	:	GetHDRModeStereo			*
 *  Parameter1	:	UINT32 ( *HDRMode)			*
 *  Returns	:	BOOL (TRUE or FALSE)			*
 *  Description	:   	Sends the extension unit command to read the HDR mode in which the camera is set. *
  **********************************************************************************************************
*/
BOOL GetHDRModeStereo(UINT32 *HDRMode)
{
	BOOL timeout = TRUE;
	int ret = 0;
	unsigned int start, end = 0;

	//Initialize the buffer	
	memset(g_out_packet_buf, 0x00, sizeof(g_out_packet_buf));
	
	//Set the Report Number
	g_out_packet_buf[1] = CAMERA_CONTROL_STEREO; 	/* Report Number */
	g_out_packet_buf[2] = GET_HDR_MODE_STEREO;
	
	ret = write(hid_fd, g_out_packet_buf, BUFFER_LENGTH);
	if (ret < 0)
	{
		perror("xunit-GetHDRMode : write failed");
		return FALSE;
	}
	else
	{
		//printf("%s(): wrote %d bytes\n", __func__,ret);
	}
	
	start = GetTickCount();
	while (timeout)
	{
		/* Get a report from the device */
		ret = read(hid_fd, g_in_packet_buf, BUFFER_LENGTH);
		
		if (ret < 0)
		{
			//perror("read");
		}
		else
		{
			//printf("%s(): read %d bytes:\n", __func__,ret);
			if(g_in_packet_buf[0] == CAMERA_CONTROL_STEREO && g_in_packet_buf[1] == GET_HDR_MODE_STEREO)
			{
				if (  g_in_packet_buf[4] == GET_SUCCESS )
				{
					*HDRMode = g_in_packet_buf[2];
					timeout = FALSE;
				}
				else
				{
					if ( g_in_packet_buf[4] == GET_FAIL )
					{
						return FALSE;
					}
				}
			}
	 	}
	 	end = GetTickCount();
		if(end - start > TIMEOUT)
		{
			printf("%s(): Timeout occurred\n", __func__);
			timeout = FALSE;
			return FALSE;
		}
	}
	return TRUE;
}


/*
  **********************************************************************************************************
 *  MODULE TYPE	:	LIBRAY API 				*
 *  Name	:	GetIMUTemperatureData			*
 *  Parameter1	:	unsigned char *	(MSBTemp)		*
 *  Parameter2	:	unsigned char *	(LSBTemp)		*
 *  Returns	:	BOOL (TRUE or FALSE)			*
 *  Description	:   	Sends the extension unit command to the UVC device		*
 *			and then device sends back the temperature of the IMU unit 	*	
  **********************************************************************************************************
*/
BOOL GetIMUTemperatureData(UINT8 *MSBTemp, UINT8 *LSBTemp)
{
	BOOL timeout = TRUE;
	int ret = 0;
	unsigned int start, end = 0;

	//Initialize the buffer	
	memset(g_out_packet_buf, 0x00, sizeof(g_out_packet_buf));
	
	//Set the Report Number
	g_out_packet_buf[1] = CAMERA_CONTROL_STEREO; 	/* Report Number */
	g_out_packet_buf[2] = GET_IMU_TEMP_DATA;	

	/* Send a Report to the Device */
	ret = write(hid_fd, g_out_packet_buf, BUFFER_LENGTH);
	if (ret < 0)
	{
		perror("xunit-GetIMUTemperatureData : write failed");
		return FALSE;
	}
	else
	{
		//printf("%s(): wrote %d bytes\n", __func__,ret);
	}
	
	start = GetTickCount();
	
	while (timeout)
	{
		/* Get a report from the device */
		ret = read(hid_fd, g_in_packet_buf, BUFFER_LENGTH);
		
		if (ret < 0)
		{
			//perror("read");
		}
		else
		{
			//printf("%s(): read %d bytes:\n", __func__,ret);
			if(g_in_packet_buf[0] == CAMERA_CONTROL_STEREO && g_in_packet_buf[1] == GET_IMU_TEMP_DATA)
			{
				if (  g_in_packet_buf[6] == GET_SUCCESS )
				{
					*MSBTemp = g_in_packet_buf[2];
					*LSBTemp = g_in_packet_buf[3];
					timeout = FALSE;
				}
				else
				{
					if ( g_in_packet_buf[6] == GET_FAIL )
					{
						return FALSE;
					}
				}
			}
	 	}
	 	end = GetTickCount();
		if(end - start > TIMEOUT)
		{
			printf("%s(): Timeout occurred\n", __func__);
			timeout = FALSE;
			return FALSE;
		}
	}
	return TRUE;	
}


/*
  **********************************************************************************************************
 *  MODULE TYPE	:	LIBRAY API 		*
 *  Name	:	bus_str			*
 *  Parameter1	:	int	(bus)		*
 *  Returns	:	const char *		*
 *  Description	:   	To convert integer bus type to string		    *	
  **********************************************************************************************************
*/
const char *bus_str(int bus)
{
	switch (bus) {
	case BUS_USB:
		return "USB";
		break;
	case BUS_HIL:
		return "HIL";
		break;
	case BUS_BLUETOOTH:
		return "Bluetooth";
		break;
	case BUS_VIRTUAL:
		return "Virtual";
		break;
	default:
		return "Other";
		break;
	}
}


/*
  **********************************************************************************************************
 *  MODULE TYPE	:	Internal API 		*
 *  Name	:	GetTicketCount		*
 *  Returns	:	unsigned int		*
 *  Description	:  	To return current time in milli seconds	   		*	
  **********************************************************************************************************
*/
unsigned int GetTickCount(void)
{
        struct timeval tv;
        if(gettimeofday(&tv, NULL) != 0)
                return 0;

        return (tv.tv_sec * 1000) + (tv.tv_usec / 1000);
}


/*
  **********************************************************************************************************
 *  MODULE TYPE	:	LIBRAY API 			    *
 *  Name	:	find_hid_device			    *
 *  Parameter1	:	char (*videobusname)		    *
 *  Returns	:	int (SUCCESS or FAILURE)	    *
 *  Description	:   	To find the first e-con's hid device connected to the linux pc	*	
  **********************************************************************************************************
*/
int find_hid_device(char *videobusname)
{
	struct udev *udev;
	struct udev_enumerate *enumerate;
	struct udev_list_entry *devices, *dev_list_entry;
	struct udev_device *dev, *pdev;
	int ret = FAILURE;
	char buf[256];
	
   	/* Create the udev object */
	udev = udev_new();
	if (!udev) {
		printf("Can't create udev\n");
		exit(1);
	}

	/* Create a list of the devices in the 'hidraw' subsystem. */
	enumerate = udev_enumerate_new(udev);
	udev_enumerate_add_match_subsystem(enumerate, "hidraw");
	udev_enumerate_scan_devices(enumerate);
	devices = udev_enumerate_get_list_entry(enumerate);
	
	/* For each item enumerated, print out its information. udev_list_entry_foreach is a macro which expands to a loop. The loop will be executed for each member in
	   devices, setting dev_list_entry to a list entry which contains the device's path in /sys. */
	udev_list_entry_foreach(dev_list_entry, devices) {
		const char *path;
		
		/* Get the filename of the /sys entry for the device and create a udev_device object (dev) representing it */
		path = udev_list_entry_get_name(dev_list_entry);
		dev = udev_device_new_from_syspath(udev, path);

		/* usb_device_get_devnode() returns the path to the device node itself in /dev. */
		//printf("Device Node Path: %s\n", udev_device_get_devnode(dev));
		
		/* The device pointed to by dev contains information about the hidraw device. In order to get information about the USB device, get the parent device with the subsystem/devtype pair of "usb"/"usb_device". This will be several levels up the tree, but the function will find it.*/
		pdev = udev_device_get_parent_with_subsystem_devtype(
		       dev,
		       "usb",
		       "usb_device");
		if (!pdev) {
			printf("Unable to find parent usb device.");
			exit(1);
		}
	
		/* From here, we can call get_sysattr_value() for each file in the device's /sys entry. The strings passed into these functions (idProduct, idVendor, serial, 			etc.) correspond directly to the files in the /sys directory which represents the USB device. Note that USB strings are Unicode, UCS2 encoded, but the strings    		returned from udev_device_get_sysattr_value() are UTF-8 encoded. */
		if(!strncmp(udev_device_get_sysattr_value(pdev,"idVendor"), "2560", 4)) {
			if(!strncmp(udev_device_get_sysattr_value(pdev, "idProduct"), "c114", 4)) {
					hid_device = udev_device_get_devnode(dev);
					udev_device_unref(pdev);
			}
		}
		else
		{
			continue;
		}

		//Open each hid device and Check for bus name here
		hid_fd = open(hid_device, O_RDWR|O_NONBLOCK);

		if (hid_fd < 0) {
			perror("find_hid_device : Unable to open device");
			continue;
		}else
			memset(buf, 0x00, sizeof(buf));

		/* Get Physical Location */
		ret = ioctl(hid_fd, HIDIOCGRAWPHYS(256), buf);
		if (ret < 0) {
			perror("find_hid_device : HIDIOCGRAWPHYS");
		}
		//check if bus names are same or else close the hid device
		if(!strncmp(videobusname,buf,strlen(videobusname))){
			ret = SUCCESS;
			hid_device_array[countHidDevices] = hid_device;
			countHidDevices++;
		}
		/* Close the hid fd */
		if(hid_fd > 0)
		{
			if(close(hid_fd) < 0) {
				printf("\nFailed to close %s\n",hid_device);
			}
		}
	}
	/* Free the enumerator object */
	udev_enumerate_unref(enumerate);
	udev_unref(udev);

	return ret;
}

//
static	PyObject *xunit_init(PyObject *self, PyObject *args)
{
	char *bus_name;
	BOOL result;
	PyObject *ret;
	  // parse arguments
	if (!PyArg_ParseTuple(args, "s", &bus_name)) {
	  return NULL;
	}
	// run the actual function
	result = InitExtensionUnit(bus_name);
	
	// build the resulting into a Python object.
	//if (result) ret = Py_True;
	//else ret = Py_False;
	ret = PyBool_FromLong((long)result);
	//free(result);
	return ret;
}

static	PyObject *xunit_deinit(PyObject *self, PyObject *args)
{
	BOOL result;
	PyObject *ret;
	// run the actual function
	result = DeinitExtensionUnit();
	
	// build the resulting into a Python object.
	//if (result) ret = Py_True;
	//else ret = Py_False;
	ret = PyBool_FromLong((long)result);
	//free(result);
	return ret;
}


static PyObject *xunit_setmanualexposure(PyObject *self, PyObject *args)
{
	int exposurevalue;
	BOOL result;
	PyObject *ret;
	if (!PyArg_ParseTuple(args, "i", &exposurevalue)) {
		return NULL;
	}
	result = SetManualExposureStereo((INT32)exposurevalue);
	//if (result) ret = Py_True;
	//else ret = Py_False;
	ret = PyBool_FromLong((long)result);
	//free(result);
	return ret;
}

static PyObject *xunit_setautoexposure(PyObject *self, PyObject *args)
{
	BOOL result;
	PyObject *ret;
	result = SetAutoExposureStereo();
	ret = PyBool_FromLong((long)result);
	//free(result);
	return ret;
}

//define module's methods
static PyMethodDef XunitMethods [] = {
	{"InitExtensionUnit", xunit_init, METH_VARARGS, "Initializes the Extension Unit"},
	{"DeinitExtensionUnit", xunit_deinit, METH_NOARGS, "Deinits the Extension Unit"},
	{"SetManualExposureStereo", xunit_setmanualexposure, METH_VARARGS, "Sets the Exposure Value passed to the camera"},
	{"SetAutoExposureStereo", xunit_setautoexposure, METH_VARARGS, "Sets the Camera to Auto Exposure"},
    {NULL,              NULL}           /* sentinel */

};

//Module struct initialization
static struct PyModuleDef xunitmodule = {
	PyModuleDef_HEAD_INIT,
	"xunit",
	NULL,
	-1,
	XunitMethods,
	NULL,
	NULL,
	NULL,
	NULL
};

//Module initialization
PyMODINIT_FUNC PyInit_xunit(void)
{
	return PyModule_Create(&xunitmodule);
}
