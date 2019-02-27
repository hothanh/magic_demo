#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <linux/input.h>
#include <linux/hidraw.h>
#include <stdbool.h>
#include <libudev.h>
#include <pthread.h>
#include <linux/videodev2.h>
#include <libv4l2.h>
#include <glib.h>
#include <fcntl.h>
#include <sys/ioctl.h>

#include <iostream>
#include <limits>

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

enum TaraRev
{
	REVISION_A = 0,
	REVISION_B = 1
};

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
#define SEE3CAM_STEREO_EXPOSURE_AUTO	(1)
#define SEE3CAM_STEREO_EXPOSURE_MIN		(10)
#define SEE3CAM_STEREO_EXPOSURE_MAX		(1000000)
#define SEE3CAM_STEREO_EXPOSURE_DEF		(8000)

/* IMU MODE */
#define IMU_ACC_GYRO_DISABLE		(0x00)
#define IMU_ACC_ENABLE				(0x01)
#define IMU_ACC_GYRO_ENABLE			(0x03)

/* ACC AXIS CONTROL */
#define IMU_ACC_X_Y_Z_ENABLE		(0x07)
#define IMU_ACC_X_ENABLE			(0x01)
#define IMU_ACC_Y_ENABLE			(0x02)
#define IMU_ACC_Z_ENABLE			(0x04)

/* ACC ODR CONTROL for Revision A*/
#define IMU_ODR_10_14_9HZ			(0x01)
#define IMU_ODR_50_59_9HZ			(0x02)
#define IMU_ODR_119HZ				(0x03)
#define IMU_ODR_238HZ				(0x04)
#define IMU_ODR_476HZ				(0x05)
#define IMU_ODR_952HZ				(0x06)

/* ACC ODR CONTROL for Rev B*/
#define IMU_ODR_12_5HZ				(0x01)
#define IMU_ODR_26HZ				(0x02)
#define IMU_ODR_52HZ				(0x03)
#define IMU_ODR_104HZ				(0x04)
#define IMU_ODR_208HZ				(0x05)
#define IMU_ODR_416HZ				(0x06)
#define IMU_ODR_833HZ				(0x07)
#define IMU_ODR_1666HZ				(0x08)


/* ACC SENSITIVITY CONTROL */
#define IMU_ACC_SENS_2G				(0x00)
#define IMU_ACC_SENS_4G				(0x02)
#define IMU_ACC_SENS_8G				(0x03)
#define IMU_ACC_SENS_16G			(0x01)

/* GYRO AXIS CONTROL */
#define IMU_GYRO_X_Y_Z_ENABLE		(0x07)
#define IMU_GYRO_X_ENABLE			(0x01)
#define IMU_GYRO_Y_ENABLE			(0x02)
#define IMU_GYRO_Z_ENABLE			(0x04)

/* GYRO SENSITIVITY CONTROL */
#define IMU_GYRO_SENS_125DPS			(0x04)
#define IMU_GYRO_SENS_250DPS			(0x00)
#define IMU_GYRO_SENS_245DPS			(0x00)
#define IMU_GYRO_SENS_500DPS			(0x01)
#define IMU_GYRO_SENS_1000DPS			(0x02)
#define IMU_GYRO_SENS_2000DPS			(0x03)

/* IMU VALUE UPDATE MODE */
#define IMU_CONT_UPDT_EN			(0x01)
#define IMU_CONT_UPDT_DIS			(0x02)

/* IMU VALUES CONTROL */
#define IMU_AXES_VALUES_MIN			(1)
#define IMU_AXES_VALUES_MAX			(65535)

/* Range of Gyro for Rev A*/
#define LSM6DS0_G_FS_245                   		(UINT8)(0x00) /* Full scale: 245 dps  */
#define LSM6DS0_G_FS_500                    	(UINT8)(0x08) /* Full scale: 500 dps  */
#define LSM6DS0_G_FS_2000                   	(UINT8)(0x18) /* Full scale: 2000 dps */


/* Range of Gyro for Rev B*/
#define LSM6DS3_G_FS_125                        (UINT8)(0x00) /* Full scale: 125 dps  */
#define LSM6DS3_G_FS_250                        (UINT8)(0x04) /* Full scale: 250 dps  */
#define LSM6DS3_G_FS_500                        (UINT8)(0x08) /* Full scale: 500 dps  */
#define LSM6DS3_G_FS_1000                       (UINT8)(0x0C) /* Full scale: 1000 dps  */
#define LSM6DS3_G_FS_2000                       (UINT8)(0x10) /* Full scale: 2000 dps */

/* Range of Accelero for Rev A*/
#define LSM6DS0_XL_FS_2G                    	(UINT8)(0x00) /* Full scale: +- 2g */
#define LSM6DS0_XL_FS_4G                    	(UINT8)(0x10) /* Full scale: +- 4g */
#define LSM6DS0_XL_FS_8G                    	(UINT8)(0x18) /* Full scale: +- 8g */
#define LSM6DS0_XL_FS_16G                   	(UINT8)(0x08) /* Full scale: +- 16g*/

/*Range of Accelero for Rev B*/
#define LSM6DS3_XL_FS_2G                        (UINT8)(0x00) /* Full scale: +- 2g */
#define LSM6DS3_XL_FS_4G                        (UINT8)(0x08) /* Full scale: +- 4g */
#define LSM6DS3_XL_FS_8G                        (UINT8)(0x0C) /* Full scale: +- 8g */
#define LSM6DS3_XL_FS_16G                       (UINT8)(0x04) /* Full scale: +- 16g*/

/* For Stereo - Tara */
/* Commands */
#define CAMERA_CONTROL_STEREO		0x78

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

#define REVISIONID					0x10
#define CAMERACONTROL_STEREO		0x78

#define SET_STREAM_MODE_STEREO		0x0B
#define GET_STREAM_MODE_STEREO		0x0C
#define GET_IMU_TEMP_DATA			0x0D
#define SET_HDR_MODE_STEREO			0x0E
#define GET_HDR_MODE_STEREO			0x0F
	
#define IMU_NUM_OF_VAL				0xFF
#define IMU_ACC_VAL					0xFE
#define IMU_GYRO_VAL				0xFD
#define INTRINSIC_FILEID			0x00
#define EXTRINSIC_FILEID			0x01
#define PCK_SIZE					  56

#define TRUE                    		1
#define FALSE                   		0

#define SDK_VERSION			"2.0.6"
#define FRAMERATE 			60
#define MASTERMODE 			1
#define TRIGGERMODE 			0
#define IOCTL_RETRY 			4
#define DEBUG_ENABLED 		1
#define DEFAULT_BRIGHTNESS 		(4.0/7.0)
#define AUTOEXPOSURE 			1 
#define DISPARITY_OPTION 		1 // 1 - Best Quality Depth Map and Lower Frame Rate - Stereo_SGBM 3 Way generic Left and Right 
					  // 0 - Low  Quality Depth Map and High  Frame Rate - Stereo_BM generic Left and Right 

//Global IMU variables.
IMUCONFIG_TypeDef				glIMUConfig;
IMUDATAINPUT_TypeDef				glIMUInput;

TaraRev g_eTaraRev;
			
BOOL						g_IsIMUConfigured = FALSE;
float						glAccSensMult = 0;
float						glGyroSensMult = 0;
			
int 						hid_fd = -1, hid_imu = -1;
int 						countHidDevices = 0;
			
unsigned char					g_out_packet_buf[BUFFER_LENGTH];
unsigned char 					g_in_packet_buf[BUFFER_LENGTH];
const char					*hid_device;
const char					*hid_device_array[2];
int DeviceID;
int ImgWidth;
int ImgHeight;
char * DeviceInfo;
//Video node properties
typedef struct _VidDevice
{
	char *device;
	char *friendlyname;
	char *bus_info;
	char *vendor;
	char *product;
	short int deviceID;
} VidDevice;

//Enumerated devices list
typedef struct _LDevices
{
	VidDevice *listVidDevices;
	int num_devices;
} LDevices;

//Stores the device instances of all enumerated devices
LDevices *DeviceInstances;

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
//extension unit init
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
		//printf("%s(): Not able to find the e-con's see3cam device\n", __func__);
		return FALSE;
	}
	

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
BOOL SetExposure(int ExposureVal)
{
	if(!SetManualExposureStereo(ExposureVal)) //Set the manual exposure
	{
		if(DEBUG_ENABLED)
			printf("SetExposure : Exposure Setting Failed\n");
		return FALSE;
	}
	return TRUE;
}

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

//Query the resolution for selected camera.
void query_resolution(int deviceid)
{
	int fd = 0;
	
	/* open the device and query the capabilities */
	if ((fd = v4l2_open(DeviceInstances->listVidDevices[deviceid].device, O_RDWR | O_NONBLOCK, 0)) < 0)
    	{
        	g_printerr("ERROR opening V4L2 interface for %s\n", DeviceInstances->listVidDevices[deviceid].device);
	        v4l2_close(fd);
       		return;
    	}

	//Query framesizes to get the supported resolutions for Y16 format.
	struct v4l2_frmsizeenum frmsize;
	frmsize.pixel_format = V4L2_PIX_FMT_Y16;
	frmsize.index = 0;
	    
	while (xioctl(fd, VIDIOC_ENUM_FRAMESIZES, &frmsize) >= 0)
	{
 		if (frmsize.type == V4L2_FRMSIZE_TYPE_DISCRETE)
		{
			CameraResolutions.push_back(cv::Size(frmsize.discrete.width, frmsize.discrete.height));
       		}
	        else if (frmsize.type == V4L2_FRMSIZE_TYPE_STEPWISE)
       		{
		        CameraResolutions.push_back(cv::Size(frmsize.stepwise.max_width, frmsize.stepwise.max_height));
      		}
	       frmsize.index++;
	}
	v4l2_close(fd);
}

bool GetDeviceIDeCon(int *DeviceID)
{
	if(DEBUG_ENABLED)
		printf("Get DeviceID eCon");
			
	short int index=-1, NoDevicesConnected = -1;
	int ResolutionID = -1;
	
	//Get the list of devices connected
	NoDevicesConnected = GetListofDeviceseCon();
	if(DEBUG_ENABLED)
		printf("\nNumber of connected devices : %d",NoDevicesConnected);

	//Check for a valid ID
	if(NoDevicesConnected <= 0)
	{
		printf("No devices connected\n\n");
		*DeviceID = -1;
		return false;
	}

	//Print the name of the devices connected
	printf("\nDevices List :\n");
	printf("----------------\n");
	for(int i = 0; i < NoDevicesConnected; i++)
	{
		printf("Device ID: %d, Device Name: %s\n",DeviceInstances->listVidDevices[i].deviceID, DeviceInstances->listVidDevices[i].friendlyname);
	}

	//User Input of the Device ID
	printf("\nEnter the Device ID to Stream : ");
	scanf("%d",*DeviceID);

	//Check for a valid ID
	if(*DeviceID < 0)
	{
		printf("Please enter a valid Device Id\n\n");
		*DeviceID = -1;
		return false;
	}

	//Finding the index of the selected device ID
	for(int i = 0; i < NoDevicesConnected; i++)
    	{
    	    	if(DeviceInstances->listVidDevices[i].deviceID == *DeviceID)
    	    	{
    	    		index = i;
    	    		break;
    	    	}		
    	}

	//Check for a valid ID
	if(index == -1)
	{
	printf("Please enter a valid Device Id\n\n");
        *DeviceID = -1;
        return false;
	}		
	

	////Check whether the selected device is Stereo
	//if(!IsStereoDeviceAvail(DeviceInstances->listVidDevices[index].product))
	//{
	//	printf("Please select a stereo camera\n\n");
	//	*DeviceID = -1;
	//	return false;
	//}

	//printf("\nResolutions Supported : \n"); 
	//printf("-------------------------\n");	
	//query_resolution(index);

	////Resolution Supported
	//for(unsigned int i = 0; i < CameraResolutions.size(); i++)
	//{
	//	printf("ID: %d, Resolution: %dx%d\n",i,CameraResolutions[i].width,CameraResolutions[i].height);
	//}

	////User Input of the Device ID
	//printf("\nEnter the Resolution ID to Stream : ");
	//scanf("%d",ResolutionID);

	//if(ResolutionID > int(CameraResolutions.size() - 1) || ResolutionID < 0) //In case of wrong selection
	//{
	//	printf("\nInvalid ResolutionID\n");
	//	*DeviceID = -1;
	//	return false;
	//}

	//Bus info of the selected device.
	DeviceInfo = DeviceInstances->listVidDevices[index].bus_info;
	return true;
}


static PyObject *InitCamera(PyObject *self, PyObject *args)
{
	//Read the device ID to stream
	GetDeviceIDeCon(&DeviceID);
	
	if(DeviceID < 0)//Check for a valid device ID
	{

		printf("InitCamera : Please select a valid device\n");
		return false;
	}

	////Open the device selected by the user.
	//_CameraDevice.open(DeviceID);
	//	
	////Camera Device
	//if(!_CameraDevice.isOpened())
	//{			
	//	printf("InitCamera : Camera opening failed\n");
	//	return false;
	//}
	//
	//if (DEFAULT_FRAME_WIDTH == ImageSize.width && DEFAULT_FRAME_HEIGHT == ImageSize.height)
	//{
	//	_CameraDevice.set(CV_CAP_PROP_FRAME_WIDTH, 752);
	//	_CameraDevice.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	//}

	////Setting up Y16 Format
	//_CameraDevice.set(CV_CAP_PROP_FOURCC, CV_FOURCC('Y', '1', '6', ' '));

	////Setting up FrameRate
	//_CameraDevice.set(CV_CAP_PROP_FPS, FRAMERATE);

	////Setting width and height
	//_CameraDevice.set(CV_CAP_PROP_FRAME_WIDTH, ImageSize.width);
	//_CameraDevice.set(CV_CAP_PROP_FRAME_HEIGHT, ImageSize.height);
	//
	////y16 format support
	//_CameraDevice.set(CV_CAP_PROP_CONVERT_RGB, 0);

	//Init the extension units
	if(!InitExtensionUnit(DeviceInfo))
	{			

		printf("InitCamera : Extension Unit Initialisation Failed\n");
		return false;
	}
	printf("DeviceInfo: %s\n",DeviceInfo);	
	//Setting up the camera in Master mode
	if(!SetStreamModeStereo(MASTERMODE))
	{			
		printf("InitCamera : Setting up Stream Mode Failed, initiating in the default mode\n");
	}

	//Setting to default Brightness
	//SetBrightness(DEFAULT_BRIGHTNESS);

	//setting up auto Exposure
	SetAutoExposureStereo();

	//Choose whether the disparity is filtered or not
	//gFilteredDisparity = FilteredDisparityMap;

	//Initialise the disparity options
	//if(!Init(GenerateDisparity))
	//{
	//	if(DEBUG_ENABLED)
	//		printf("InitCamera : Camera Matrix Initialisation Failed\n");
	//	return false;
	//}

	//Mat creation
	//InterleavedFrame.create(ImageSize.height, ImageSize.width, CV_8UC2);

	return PyLong_FromLong(*DeviceID);
}

//define module's methods
static PyMethodDef TaraMethods [] = {
    {"InitCamera", InitCamera, METH_NOARGS, "Initializes the Extension Unit"},
    {NULL,              NULL}           /* sentinel */

};

//Module struct initialization
static struct PyModuleDef taramodule = {
	PyModuleDef_HEAD_INIT,
	"tara",
	NULL,
	-1,
	TaraMethods,
	NULL,
	NULL,
	NULL,
	NULL
};

//Module initialization
PyMODINIT_FUNC PyInit_tara(void)
{
	return PyModule_Create(&taramodule);
}
