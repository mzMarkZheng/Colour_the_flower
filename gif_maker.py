import imageio
from os import listdir
from os.path import isfile, join

from PIL import _imaging
from PIL import Image, ImageOps, ImageFont, ImageDraw

mypath = "E:\\Repo\\ML\\Deep Neuron\\colour-the-flower\\Example_results"
onlyfiles = [f for f in listdir(
    mypath) if isfile(join(mypath, f))]

images = []
filenames = onlyfiles

filenames.sort(key=lambda x: int(x[5:-4]))

with imageio.get_writer('/results.gif', mode='I', duration=0.2) as writer:
    for filename in filenames:
        print(filename)
        image = imageio.imread("Example_results\\" + filename)
        writer.append_data(image)
