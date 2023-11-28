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
