# Data description

The following directory should be setup to contain the data used by the project. Because of its voluminous size, the data is not hosted in this github repo.

Instead, you can find all the "raw" data (pictures), as well as the corresponding pre-processed version [here](TODO). Since the size images is big, you can selectively downloa different subfolders depending on what you want to test.

As such, to recreate every aspect of the project, the *complete* directory structure should look like this:

```
data
├── prepro              <-- The pre-processed version of the raw (see here for more info)
    ├───...
├── raw                 <-- The raw data from various of the camera captures (.RAF raws, and .HIF for the initial data acquisition comparison)
    ├───...
├── datasets/          # Training datasets
├── checkpoints/       # Model checkpoints
└── reproduction        <-- Files used for reproduction of results (e.g.: darktable styles, presets)
```

The analog pictures have been captured at f/8, (film) ISO 400. The exposure time for the different pictures can be found in the table below.

| Filename | Exposure Time (s) | Scene Description
|----------|---------------|---------------|
| DSCF7039.tif | 1/30 |Color checkerboard under OPT illumination|
| DSCF7040.tif | 1/30 |Color checkerboard under D65 illumination|
| DSCF7041.tif | 1/30 |Color checkerboard under House illumination|
| DSCF7042.tif | 1/30 |Color checkerboard under City illumination|
| DSCF7044.tif | 1/30 |Noise test chart, back-illuminated|
| DSCF7045.tif | 1/30 |Color gradient (color -> gray) chart, back-illuminated|
| DSCF7048.tif | 1/30 |Color gradient (color -> white) chart, back-illuminated|
| DSCF7051.tif | 1/30 |Scala stairs in the BC building (indoors, daytime) |
| DSCF7054.tif | 1/500 |Houses amongst trees (outdoors, cloudy, daytime)|
| DSCF7057.tif | 1/60 | Zack and Kevan sitting on a hammack (outdoors, cloudy day, daytime)|
| DSCF7060.tif | 1/125 | A red jacket on green grass (outdoors, cloudy, daytime)|
| DSCF7063.tif | 1/500 | The red EPFL logo, with the Rolex Learning Center and mountains in the backround (outdoors, cloudy, daytime)|
| DSCF7066.tif | 1/125 |A building with colored strips (outdoors, cloudy, daytime)|
| DSCF7069.tif | 1/60 |Orange building outside of a window, viewed from inside (indoors; outdoors, cloudy, daytime)|
| DSCF7072.tif | 1/2 |Eren on a green background (indoors)|
| DSCF7075.tif | 1/2 |Victor on a green backgound (indoors)|
| DSCF7143.tif | 1/300 |Grass and sky picture (outdoors, small cloud cover, daytime)|
| DSCF7144.tif | 1/500 |The Pelican statue, with lake Leman in the background and grass in the foreground (outdoors, sunny, daytime, close to sunset)|
| DSCF7145.tif | 1/2000 |Lake Leman with the sunset and trees (outdoors, sunny, sunset)|
| DSCF7146.tif | 1/500 |Lausanne from afar, with Lake Leman in the foreground (outdoors, sunny, sunset)|
| DSCF7147.tif | 1/125 |Tree branches (outdoors, sunny, sunset)|
| DSCF7148.tif | 1/1000 |Coast, Lake Leman, and yellow sky (outdoors, sunny, peak sunset)|
| DSCF7149.tif | 1/2000 |Beach coastline, Lake Leman, trees and yellow sky (outdoors, sunny, peak sunset)|
| DSCF7150.tif | 1/60 |Garden with cherry trees (outdoors, sunny, end of sunset)|
| DSCF7151.tif | 1/8 |Close-up of red, green and yellow-colored leaves on a fence  (outdoors, sunny, end of sunset)|
| DSCF7152.tif & DSCF7153.tif | 1 |Red and Blue car (outdoors, dusk/beginning of night)|
| DSCF7155.tif | 1/4 |Grafitis on a wall, with a brigth-red LED parking barrier (outdoors, nighttime)|
| DSCF7160.tif | 1/60 |Bessières bridge with Lausanne cathedral in the background (outdoors, nighttime)|
| DSCF7157.tif | 1 |Illuminated Lausanne cathedral (outdoors, nighttime)|
| DSCF7156.tif | 0.5 |Place de la Riponne, with blue and red advertisement pannels (outdoors, nighttime)|
| DSCF7159.tif | 1/8 |Lausanne metro at Flon, with colored columns (indoors, nighttime)|

After taking an analog picture, the digital "baseline" was captured with the same settings (f-score, ISO, exposure time), matching the focus point as closely as possible. 


## Data Filtering

The camera used to capture the pictures is a Fujifilm X-T50. Its RAW file format is a proprietary FUJIFILMCCD-RAW (.ARF) file format. The camera is designed to capture multiple pictures at once, at 3 different ISO sensitivities, for potential post-processing HDR. As such, for reproducibility and closest comparison to the analog picture, we must only select the picture with ISO = 400. Due to the closed-source nature of .ARF, it is complicated to find where to search for the makernote inside the RAW file to extract the metadata. Most tools and specs found online (e.g.: [this](https://libopenraw.freedesktop.org/formats/raf/) or [this](https://github.com/franzwong/fujifilm-raf-reader?tab=readme-ov-file)) assume that the file is encoded according to a reverse-engineered specification [unpublished in 2006](https://web.archive.org/web/20090213050537/http://crousseau.free.fr/imgfmt_raw.htm), or do not support our camera/Fujifilm at all (e.g.: [ExifRead](https://pypi.org/project/ExifRead/), or the industry standard [LibRaw](https://www.libraw.org/supported-cameras) which, while it can load the RAW picture, does not offer metadata information). We have therefore used [exiftool](https://exiftool.org/index.html) (through [its python wrapper](https://sylikc.github.io/pyexiftool/) for easier scripting) which although it doesn't cite its source for extracting information from the .ARF file, outputs readings which are coherent with Fujifilm's recommended [RAW FILE CONVERTER](https://www.fujifilm-x.com/global/support/download/software/raw-file-converter-ex-powered-by-silkypix/).

