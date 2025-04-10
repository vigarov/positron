# Data description

The following directory should be setup to contain the data used by the project. Because of its voluminous size, the data is not hosted in this github repo.

Instead, you can find all the "raw" data (pictures), as well as the corresponding pre-processed version [here](TODO). Since the size images is big, you can selectively downloa different subfolders depending on what you want to test.

As such, to recreate every aspect of the project, the *complete* directory structure should look like this:

```
data
├───baselines           <-- The baselines used to compare our approach
    ├───...
├───prepro              <-- The pre-processed version of the raw (see here for more info)
    ├───...
├───raw                 <-- The raw data from various of the camera captures (.RAF raws, and .HIF for the initial data acquisition comparison)
    ├───...
└───reproduction        <-- Files used for reproduction of results (e.g.: darktable styles, presets)
```

The analog pictures have been captured at f/8, (film) ISO 400. The exposure time for the different pictures can be found in the table below.

TABLE TODO

After taking an analog picture, the digital "baseline" was captured with the same settings (f-score, ISO, exposure time), matching the focus point as closely as possible. 


## Data Filtering

The camera used to capture the pictures is a Fujifilm X-T50. Its RAW file format is a proprietary FUJIFILMCCD-RAW (.ARF) file format. The camera is designed to capture multiple pictures at once, at 3 different ISO sensitivities, for potential post-processing HDR. As such, for reproducibility and closest comparison to the analog picture, we must only select the picture with ISO = 400. Due to the closed-source nature of .ARF, it is complicated to find where to search for the makernote inside the RAW file to extract the metadata. Most tools and specs found online (e.g.: [this](https://libopenraw.freedesktop.org/formats/raf/) or [this](https://github.com/franzwong/fujifilm-raf-reader?tab=readme-ov-file)) assume that the file is encoded according to a reverse-engineered specification [unpublished in 2006](https://web.archive.org/web/20090213050537/http://crousseau.free.fr/imgfmt_raw.htm), or do not support our camera/Fujifilm at all (e.g.: [ExifRead](https://pypi.org/project/ExifRead/), or the industry standard [LibRaw](https://www.libraw.org/supported-cameras) which, while it can load the RAW picture, does not offer metadata information). We have therefore used [exiftool](https://exiftool.org/index.html) (through [its python wrapper](https://sylikc.github.io/pyexiftool/) for easier scripting) which although it doesn't cite its source for extracting information from the .ARF file, outputs readings which are coherent with Fujifilm's recommended [RAW FILE CONVERTER](https://www.fujifilm-x.com/global/support/download/software/raw-file-converter-ex-powered-by-silkypix/).

