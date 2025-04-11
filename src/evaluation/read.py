import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Percorsi ai file
label_path = "data/RETOUCH_dataset/RETOUCH_PROCESSED_NNUNET/nnUNet_raw/Dataset003_Total/labelsTe/TRAIN_021.nii.gz"
image_path = "data/RETOUCH_dataset/RETOUCH_PROCESSED_NNUNET/nnUNet_raw/Dataset003_Total/imagesTe/TRAIN_021_0000.nii.gz"
prediction_path = "data/RETOUCH_dataset/RETOUCH_PROCESSED_NNUNET/nnUNet_prediction/prediction_2d/TRAIN_021.nii.gz"

# Carica i dati
image = nib.load(image_path).get_fdata()
label = nib.load(label_path).get_fdata() 
prediction = nib.load(prediction_path).get_fdata()

# Ottieni dimensioni
num_slices = image.shape[2]
slice_idx = num_slices // 2  # Slice iniziale (centrale)

# Crea la figura e i subplot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
plt.subplots_adjust(bottom=0.2)

# Mostra immagini iniziali con imshow (senza ridisegnare ogni volta)
img1 = ax1.imshow(image[:, :, slice_idx], cmap="gray")
ax1.set_title("Immagine Originale")
ax1.axis("off")

img2 = ax2.imshow(image[:, :, slice_idx], cmap="gray")
overlay2 = ax2.imshow(label[:, :, slice_idx], cmap="jet", alpha=0.5)
ax2.set_title("Ground Truth (Label)")
ax2.axis("off")

img3 = ax3.imshow(image[:, :, slice_idx], cmap="gray")
overlay3 = ax3.imshow(prediction[:, :, slice_idx], cmap="jet", alpha=0.5)
ax3.set_title("Segmentazione Predetta")
ax3.axis("off")

# Funzione per aggiornare le immagini senza ridisegnare tutto
def update(slice_idx):
    slice_idx = int(slice_idx)  # Converti in intero
    img1.set_data(image[:, :, slice_idx])
    img2.set_data(image[:, :, slice_idx])
    overlay2.set_data(label[:, :, slice_idx])
    img3.set_data(image[:, :, slice_idx])
    overlay3.set_data(prediction[:, :, slice_idx])
    fig.canvas.draw_idle()  # Aggiorna la figura

# Callback per la tastiera
def on_key(event):
    global slice_idx
    if event.key == "right":
        slice_idx = min(slice_idx + 1, num_slices - 1)
    elif event.key == "left":
        slice_idx = max(slice_idx - 1, 0)
    update(slice_idx)
    slider.set_val(slice_idx)  # Aggiorna anche lo slider

# Aggiungi lo slider per scorrere le slice
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor="lightgray")
slider = Slider(ax_slider, "Slice", 0, num_slices - 1, valinit=slice_idx, valstep=1)
slider.on_changed(lambda val: update(int(val)))

# Assegna la funzione alla pressione dei tasti
fig.canvas.mpl_connect("key_press_event", on_key)

plt.show()
