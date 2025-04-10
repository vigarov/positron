# Given a directory with many raw files, opens them, analyses their metadata, 
#       finds the set of overlapping picture characteristics (e.g.: same ISOs, same aperture, same exposure time, ...), 
#       and prompts user which of them to select


import exiftool
from pathlib import Path
import argparse
import shutil
# From https://en.wikipedia.org/wiki/Raw_image_format : 
RAW_EXTENSIONS = [".3fr", ".ari", ".arw", ".bay", ".braw", ".crw", ".cr2", ".cr3", ".cap", ".data", ".dcs", ".dcr", ".dng", ".drf", ".eip", ".erf", ".fff", ".gpr", ".iiq", ".k25", ".kdc", ".mdc", ".mef", ".mos", ".mrw", ".nef", ".nrw", ".obm", ".orf", ".pef", ".ptx", ".pxn", ".r3d", ".raf", ".raw", ".rwl", ".rw2", ".rwz", ".sr2", ".srf", ".srw", ".tif", ".x3f"]

def parse_args():
    DEFAULT_PATH = "data/raw/Data Acquisition/"
    DEFAULT_OUT = "data/prepro/fs/"
    DEFAULT_ISO = 400

    def path_converter(path_str, supposed_dir = True) -> Path:
        path = Path(path_str) if path_str else None
        if path:
            if supposed_dir and not path.exists():
                path.mkdir(parents=True)
        return path
    
    parser = argparse.ArgumentParser(description="Keeps only the raw files with the selected ISO (--iso flag). Deletes the files if `-d` is set.")
    parser.add_argument("--path", type=path_converter, default=path_converter(DEFAULT_PATH), help=f"Path to input data (default: {DEFAULT_PATH})")
    parser.add_argument("--iso", type=int, default=DEFAULT_ISO, help=f"ISO value (default: {DEFAULT_ISO})")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--output_copy", type=path_converter, metavar="DIR", 
                             help=f"Copy correct files to specified output directory. Mutually exclusive with -d")
    output_group.add_argument("-o", action="store_const", const=path_converter(DEFAULT_OUT), dest="output_copy",
                             help=f"Shorthand to copy files to default output directory: {DEFAULT_OUT}")
    output_group.add_argument("-d", "--delete", action="store_true", 
                             help="Delete incorrect files instead of copying correct ones")
    args = parser.parse_args()

    return args

def main(args):
    files_to_keep = []
    with exiftool.ExifToolHelper() as et:
        for file in args.path.glob("*"):
            file : Path = file
            if file.suffix.lower() in RAW_EXTENSIONS:
                metadata = et.get_metadata(file)[0]
                if metadata["EXIF:ISO"] == args.iso:
                    if args.delete:
                        file.unlink()
                    elif args.output_copy is not None:
                        shutil.copy2(file,(args.output_copy/file.name) )
                    else:
                        files_to_keep.append(file.as_posix())
    if not args.delete:
        print(files_to_keep)

if __name__ == "__main__":
    main(parse_args())