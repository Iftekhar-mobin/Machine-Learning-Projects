import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import cv2

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Some constants


#Loads all the scans in the given path
#Add SliceThickness with the meta data
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

#resize the given slices to 50px*50px
#Convert to Hounsfield Unit

def get_pixels_hu(slices):
    image = np.stack([cv2.resize(np.array(each_slice.pixel_array),(50,50)) for each_slice in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)

    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

#Resample the whole 3d image in a fixed 1mm*1mm*1mm voxel size
def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

#segmentation the lungs area
#all the pixels below -320 are cancelled as they are not important for us
#all the values above 600 are cancelled as they are not imortant
def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8) + 1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0, 0]

    # Fill the air around the person
    binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image

#plots a 3d image
def plot_3d(image, threshold=-300):

# Position the scan upright,
# so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces = measure.marching_cubes_classic(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

# Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

#input folder of the dataset and load the labels from the dataset label
INPUT_FOLDER = 'C:/stage1/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()
labels_df = pd.read_csv('D:/stage1_labels.csv', index_col=0)
processed_data=[]
count=0
#loads and Calls all the functions
#add labels with the final preprocessed data
for i,patient in enumerate(patients[0]):

    first_patient = load_scan(INPUT_FOLDER + patients[i])

    first_patient_pixels = get_pixels_hu(first_patient)
    pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1, 1, 1])
    #plot_3d(pix_resampled,0)
    [x,y,z]=pix_resampled.shape
    print("after resample",pix_resampled.shape)
    pix_resampled=np.lib.pad(pix_resampled,(500-x,0),'constant',constant_values=(0,0))
    print("final_resampled",pix_resampled.shape)
    #print("shape after resampling",pix_resampled.shape)
    #print(spacing)


    segmented_lungs = segment_lung_mask(pix_resampled, False)
    #plot_3d(segmented_lungs,0)
    segmented_lungs_fill = segment_lung_mask(pix_resampled, True)
    #plot_3d(segmented_lungs_fill,0)
    label=labels_df.at[patients[i],'cancer']
    #print(label)
    if label==1:
       label=np.array([0,1])
    elif label==0:
        label=np.array([1,0])
    #print(segmented_lungs_fill.pixel_array)

    #if i>0:
        #break

    #plot_3d(segmented_lungs,0)
    #new_image=cv2.resize(segmented_lungs,(200,200,100))
    #plot_3d(new_image,0)
    count+=1
    img_data=segmented_lungs_fill-segmented_lungs
    #plot_3d(img_data,0)
    print ('Finished:',count)
    #append the 3d image of a patient with the list
    processed_data.append([img_data,label])
#saves the preprocessed data
np.save("D:/muchdata-{}-{}-{}.npy".format(50,50,20),processed_data)