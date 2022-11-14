import os
import sys
import random
import time
from tkinter import *
from tkinter import filedialog

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
from pydicom import dcmread

sitk.ProcessObject.SetGlobalDefaultThreader('Pool')

folder_path_data = {
    "fix_ser": None,
    "mov_ser": None,
    "save_DCM": None
}


class App:
    def __init__(self):
        self.window = Tk()
        self.window.title("r-reg [FOX Inc.]")
        self.window.geometry('824x242')
        self.window.minsize(824, 242)
        self.window.maxsize(824, 242)
        self.window.grid_rowconfigure(7)
        self.window.grid_columnconfigure(3)
        self.check_var = IntVar()

        if getattr(sys, 'frozen', False):
            self.image_file = Image.open(os.path.join(sys._MEIPASS, "./bg.png"))
        else:
            self.image_file = Image.open("./bg.png")

        # self.image_file = Image.open("./bg.png")
        self.photo = ImageTk.PhotoImage(self.image_file)
        self.canvas = Canvas(self.window, height=240, width=712)
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas.grid(column=3, rowspan=7, sticky='nw', ipadx=15)

        self.btn = Button(self.window, text="Fixed series", width=10, command=self.fix_ser)
        self.btn.grid(columnspan=2, row=0, sticky='ew', ipadx=15)
        self.btn = Button(self.window, text="Moving series", width=10, command=self.mov_ser)
        self.btn.grid(columnspan=2, row=1, sticky='ew', ipadx=15)
        self.lbl = Label(self.window, text="bins:", font=("Arial Bold", 10), anchor='e', width=1)
        self.lbl.grid(column=0, row=2, sticky='ew', ipadx=15)
        self.ent_bins = Entry(self.window, width=5)
        self.ent_bins.insert(0, "30")
        self.ent_bins.grid(column=1, row=2, sticky="ew")
        self.btn = Button(self.window, text="Registration !", width=10, command=self.transform)
        self.btn.grid(columnspan=2, row=3, sticky='ew', ipadx=15)
        self.check = Checkbutton(self.window, text="reverse DICOM", variable=self.check_var)
        self.check.grid(columnspan=2, row=4, sticky='ew', ipadx=15)
        self.btn = Button(self.window, text="save DICOM", width=10, command=self.save_DCM)
        self.btn.grid(columnspan=2, row=5, sticky='ew', ipadx=15)
        self.btn = Button(self.window, text='Close', width=10, command=lambda: self.window.destroy())
        self.btn.grid(columnspan=2, row=6, sticky='ew', ipadx=15)

        self.window.mainloop()

    def fix_ser(self):
        global folder_path_data
        folder_path_data["fix_ser"] = filedialog.askdirectory()
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(folder_path_data["fix_ser"])
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        sitk.WriteImage(image, f'fix.nii')

    def mov_ser(self):
        global folder_path_data
        folder_path_data["mov_ser"] = filedialog.askdirectory()
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(folder_path_data["mov_ser"])
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        sitk.WriteImage(image, f'move.nii')

    def transform(self):
        global folder_path_data
        if (folder_path_data["fix_ser"] is not None) and (folder_path_data["mov_ser"] is not None):
            ent_beens_n = int(self.ent_bins.get())
            fixed0 = sitk.ReadImage("./fix.nii", sitk.sitkFloat32)
            moving0 = sitk.ReadImage("./move.nii", sitk.sitkFloat32)
            initial_transform = sitk.CenteredTransformInitializer(fixed0,
                                                                  moving0,
                                                                  sitk.Euler3DTransform(),
                                                                  sitk.CenteredTransformInitializerFilter.GEOMETRY)
            moving_resampled = sitk.Resample(moving0, fixed0, initial_transform, sitk.sitkLinear, 0.0,
                                             moving0.GetPixelID())
            registration_method = sitk.ImageRegistrationMethod()
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=ent_beens_n)
            registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
            registration_method.SetMetricSamplingPercentage(0.3)
            registration_method.SetInterpolator(sitk.sitkLinear)
            registration_method.SetOptimizerAsLBFGSB()
            registration_method.SetOptimizerScalesFromPhysicalShift()
            registration_method.SetInitialTransform(initial_transform, inPlace=False)
            final_transform = registration_method.Execute(sitk.Cast(fixed0, sitk.sitkFloat32),
                                                          sitk.Cast(moving0, sitk.sitkFloat32))
            moving_resampled = sitk.Resample(moving0, fixed0, final_transform, sitk.sitkLinear, 0.0,
                                             moving0.GetPixelID())
            sitk.WriteImage(moving_resampled, f"move_r.nii")

            fixed_array = sitk.GetArrayFromImage(fixed0)
            fixed_array = fixed_array.transpose([1, 2, 0])
            moving_array = sitk.GetArrayFromImage(moving_resampled)
            moving_array = moving_array.transpose([1, 2, 0])
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            fig.set_tight_layout('pad')
            plt.subplots_adjust(wspace=0, hspace=0)
            axs[0].imshow(moving_array[:, :, moving_array.shape[2] // 2], vmin=0,
                          vmax=np.mean(moving_array[moving_array > 10]) * 2, cmap="Greens")
            axs[0].imshow(fixed_array[:, :, fixed_array.shape[2] // 2], vmin=0,
                          vmax=np.mean(fixed_array[moving_array > 0]) * 3, cmap="Reds", alpha=0.5)
            axs[0].axis('off')
            axs[0].set(aspect='auto')
            axs[1].imshow(moving_array[:, moving_array.shape[1] // 2, :], vmin=0,
                          vmax=np.mean(moving_array[moving_array > 10]) * 2, cmap="Greens")
            axs[1].imshow(fixed_array[:, fixed_array.shape[1] // 2, :], vmin=0,
                          vmax=np.mean(fixed_array[moving_array > 0]) * 3, cmap="Reds", alpha=0.5)
            axs[1].axis('off')
            axs[1].set(aspect='auto')
            axs[2].imshow(moving_array[moving_array.shape[0] // 2, :, :], vmin=0,
                          vmax=np.mean(moving_array[moving_array > 10]) * 2, cmap="Greens")
            axs[2].imshow(fixed_array[fixed_array.shape[0] // 2, :, :], vmin=0,
                          vmax=np.mean(fixed_array[moving_array > 0]) * 3, cmap="Reds", alpha=0.5)
            axs[2].axis('off')
            axs[2].set(aspect='auto')
            fig.set_facecolor("White")
            plt.savefig(f"temp.png", bbox_inches="tight")
            self.image_file = Image.open(f"temp.png")
            self.image_file = self.image_file.resize((712, 240), Image.ANTIALIAS)
            self.photo = ImageTk.PhotoImage(self.image_file)
            self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
            self.canvas.grid(column=3, rowspan=7, sticky='nw', ipadx=15)

    def save_DCM(self):
        global folder_path_data
        folder_path_data["save_DCM"] = filedialog.askdirectory()
        map_im = sitk.ReadImage("./move_r.nii", sitk.sitkInt32)
        path = folder_path_data["save_DCM"]
        path = os.path.normpath(path)

        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(folder_path_data["fix_ser"])
        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        dcm_list_sorted = list(dicom_names)
        dcm_list_sorted.sort()
        dcm_list_sorted = tuple(dcm_list_sorted)
        ds = []
        for name in dcm_list_sorted:
            with open(name, "rb") as f_in:
                dsTemp = dcmread(f_in)
            del dsTemp.SeriesInstanceUID
            del dsTemp.SOPInstanceUID
            del dsTemp.file_meta.MediaStorageSOPInstanceUID
            ds.append(dsTemp)

        SeriesNumber = ds[0].SeriesNumber + 100
        seriesInstanceUIDSrc = self.generateUID(1, SeriesNumber, 0)
        seriesInstanceUIDColor = self.generateUID(1, SeriesNumber, 1)

        for slice_number in range(map_im.GetSize()[2]):
            dsExport = ds[slice_number]
            dsExport.InstanceNumber = slice_number + 1
            dsExport.SeriesInstanceUID = seriesInstanceUIDSrc
            dsExport.SOPInstanceUID = self.generateUID(1, SeriesNumber, slice_number)
            dsExport.file_meta.MediaStorageSOPInstanceUID = dsExport.SOPInstanceUID

            dsExport.SamplesPerPixel = 1
            dsExport.PhotometricInterpretation = 'MONOCHROME2'
            dsExport.BitsAllocated = 16
            dsExport.BitsStored = 16
            dsExport.HighBit = 15
            dsExport.PixelRepresentation = 0
            if self.check_var.get() == 1:
                dsExport.Rows = map_im[:, :, slice_number].GetWidth()
                dsExport.Columns = map_im[:, :, slice_number].GetHeight()
            else:
                dsExport.Rows = map_im[:, :, slice_number].GetHeight()
                dsExport.Columns = map_im[:, :, slice_number].GetWidth()
            dsExport.RescaleIntercept = 0
            Data16 = np.uint16(map_im[:, :, slice_number])
            dsExport.PixelData = Data16.tobytes()

            dsExport.SeriesDescription = 'FOX Inc.'
            dsExport.ImageComments = f'r_{SeriesNumber}_{slice_number + 1}'
            name = path + f'\\r_{SeriesNumber}_{slice_number + 1}.dcm'
            dsExport.save_as(name)

    def generateUID(self, study_number, series_number, image_number):
        uid_str = "1.2.826.0.1.3680043.10.1007."  # get from https://www.medicalconnections.co.uk/
        date_fmt = '%Y%m%d%H%M%S'
        dateTime = time.strftime(date_fmt)
        safe_counter = random.randint(10000, 999999)
        uid_str = uid_str + f'{study_number}' + '.' + f'{series_number}' + '.' + f'{image_number}' + '.' + dateTime + '.' + f'{safe_counter}'
        return uid_str


app = App()
