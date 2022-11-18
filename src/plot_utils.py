from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib


# Colormap (same as in the paper)
_cm = matplotlib.cm.get_cmap('tab20')
def_colors = _cm.colors
cus_colors = ['k'] + [def_colors[i] for i in range(1, 19)]+['w']
cmap = ListedColormap(colors=cus_colors, name='agri', N=20)


label_names =   [
                    "Background", "Meadow", "Soft winter wheat",
                    "Corn", "Winter barley", "Winter rapeseed",
                    "Spring barley", "Sunflower", "Grapevine",
                    "Beet", "Winter triticale", "Winter durum wheat",
                    "Fruits, vegetables, flowers", "Potatoes",
                    "Leguminous fodder", "Soybeans", "Orchard",
                    "Mixed cereal", "Sorghum", "Void label"
                ]


# def get_rgb(x, swapaxes=True):
#     """Gets an observation from a time series and normalises it for visualisation."""
#     im = np.array([x[2], x[1], x[0]])
#     mx = im.max(axis=(1, 2))
#     mi = im.min(axis=(1, 2))
#     im = (im - mi[:, None, None]) / (mx - mi)[:, None, None]
#     if swapaxes:
#         im = im.swapaxes(0, 2).swapaxes(0, 1)
#         im = np.clip(im, a_max=1, a_min=0)
#     return im

def get_rgb(x, swapaxes=True):
    """Gets an observation from a time series and normalises it for visualisation."""
    im = np.array([x[2], x[1], x[0]])
    im[im > 7500] = 7500
    im = im**.8
    im = im / im.max()
    if swapaxes:
        im = im.swapaxes(0, 2).swapaxes(0, 1)
    return im


def get_plt_patch(label, geometry):
    color = cmap.colors[label]
    poly = matplotlib.patches.Polygon(np.array(geometry.__geo_interface__['coordinates']).squeeze(),
                                      fill=True, alpha=.5, color=color)
    poly_ext = matplotlib.patches.Polygon(np.array(geometry.__geo_interface__['coordinates']).squeeze(),
                                          fill=False, alpha=1, linewidth=1.5, color=color)
    return poly, poly_ext
