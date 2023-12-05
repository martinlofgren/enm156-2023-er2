# ENM156 2023 group 12 - Gender classification of bees (ER2)

## Converting PASCAL VOC XML annotation file to YOLO txt file

The script `src/xml-to-yolo.py` can be used to convert annotation files
from the PASCAL VOC XML file format to the YOLO txt file format. It is
used like so:

```sh
$ python3 src/xml-to-yolo.py some-annotation-file.xml
Converted some-annotation-file.xml and wrote to some-annotation-file.txt
$ cat some-annotation-file.txt
0 0.786 0.851 0.190 0.292
```

To use it on multiple files at once, it can be scripted (in *nix:es):

```sh
$ for f in *.xml; do python src/xml-to-yolo.py $f; done
```

## Generating synthetic data

`src/generate_synthetic_images.py` is a quick n' dirty script for generating
synthetic data containing (drone) bees on various backdrops. It assumes
background images and drone images in specific locations, and outputs images
and yolo annotation `.txt` files in the same directory as the script is run.

### Known bugs

When trying to find a place to put a bee without overlapping previously placed
bees, the script might block forever if there's no place available. There's no
detection of this in the script, so in this case, the script needs to be
terminated. One way to handle this would be to use the
[timeout](https://man7.org/linux/man-pages/man1/timeout.1.html) when invoking
the script. So, if you want to generate 1000 images, one could do something
like this:

```sh
$ for i in $(seq 1000); do timeout --signal=SIGINT 2 python3 src/generate-synthetic-images.py; done
```
