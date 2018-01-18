import numpy as np
from scipy.misc import imsave
from matplotlib import pyplot as plt

def plot_piano_roll(track, mode='beat', beat_resolution=24, grid_width=.5,
                    label=True, top=False, right=False, **kwargs):
    to_plot = track.pianoroll.T / 127.0
    top = track.lowest + track.pianoroll.shape[1] - 1
    extent = (0, track.pianoroll.shape[0], track.lowest, top)

    fig, ax = plt.subplots()

    ax.imshow(to_plot, cmap='Blues', origin='lower', extent=extent,
              interpolation='none', vmin=0, vmax=1, **kwargs)

    if mode == 'beat':
        num_beat = track.pianoroll.shape[1]//beat_resolution - 1
        xticks_major = beat_resolution * np.arange(0, num_beat)
        xticks_minor = beat_resolution * (0.5 + np.arange(0, num_beat))
        xtick_labels = np.arange(1, 1 + num_beat)
        ax.set_xticks(xticks_major)
        ax.set_xticklabels('')
        ax.set_xticks(xticks_minor, minor=True)
        ax.set_xticklabels(xtick_labels, minor=True)

    ytick_first = track.lowest + 12 - track.lowest%12
    ytick_last = top - track.lowest%12
    yticks = np.arange(ytick_first, ytick_last, 12)
    ytick_labels = ['C{}'.format(octave-2)
                    for octave in range(ytick_first//12, ytick_last//12+1)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)

    ax.grid(axis='both', linestyle=':', color='k', linewidth=grid_width)

    ax.tick_params(direction='in', top=top, right=right)
    ax.tick_params(axis='x', which='minor', width=0)

    if label:
        ax.set_xlabel('time (beat)')
        ax.set_ylabel('pitch')

    plt.show()



def save_bars(bars, size, file_path, boarder=3, boarder_color='w'):
    merged = merge(to_image_np(bars), size, boarder, boarder_color)
    scipy.misc.imsave(file_path, merged)

def merge(images, size, boarder=3, boarder_color='w'):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h*size[0] + boarder*(size[0]-1), w*size[1] + boarder*(size[1]-1), 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        add_h = boarder if j < size[0] else 0
        add_w = boarder if i < size[1] else 0
        img[j*(h+add_h):j*(h+add_h)+h, i*(w+add_w):i*(w+add_w)+w, :] = image

    for i in range(1,size[1]):
        img[:,i*(w+3)-3:i*(w+3)] = [1.0, 1.0, 1.0] if boarder_color=='w' else [-1.0, -1.0, -1.0]
    for j in range(1,size[0]):
        img[j*(h+3)-3:j*(h+3),:] = [1.0, 1.0, 1.0] if boarder_color=='w' else [-1.0, -1.0, -1.0]
    return img

def to_image_np(bars):
    colormap = np.array([[1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.],
                         [1., .5, 0.],
                         [0., .5, 1.]])
    recolored_bars = np.matmul(bars.reshape(-1, 5), colormap).reshape((bars.shape[0], bars.shape[1], bars.shape[2], 3))
    # recolored_bars = np.zeros((bars.shape[0], bars.shape[1], bars.shape[2], 3))
    # for track_idx in range(bars.shape[-1]):
    #     recolored_bars = recolored_bars + bars[..., track_idx][:, :, :, None]*colormap[track_idx][None, None, None, :]
    return np.flip(np.transpose(recolored_bars, (0, 2, 1, 3)), axis=1)

def save_bars(bars, size, file_path, boarder=3, boarder_color='w'):
    merged = merge(to_image_np(bars), size, boarder, boarder_color)
    scipy.misc.imsave(file_path, merged)