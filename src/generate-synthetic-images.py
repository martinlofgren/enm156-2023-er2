# Generate synthetic data by putting bees on other images

from PIL import Image, ImageDraw
import random
import os
import uuid
from pathlib import Path

BG_DIR = 'datasets/synthetic-raw/bg-images'
FG_DIR = 'datasets/synthetic-raw/bee/drones'
DRONE_LABEL = 1

def intersects(this, that):
    '''Detect if one rectangle intersects another'''
    return not (
        this['x'] + this['width'] < that['x']
        or that['x'] + that['width'] < this['x']
        or this['y'] + this['height'] < that['y']
        or that['y'] + that['height'] < this['y']
    )

def random_file(p):
    '''Return full path to a random file from a directory'''
    f = random.choice([x for x in os.listdir(p) if os.path.isfile(os.path.join(p, x))])
    return os.path.join(p, f)

def add_bee(ctx):
    '''Add a bee at a random position, scaled and rotated, and update context with box position and size'''
    fg_file = random_file(FG_DIR)
    fg_image = Image.open(fg_file)

    rotation = random.uniform(0, 360)
    fg_image = fg_image.rotate(rotation, expand=1)

    scale = random.uniform(0.75, 1.5)
    fg_width, fg_height = fg_image.size
    fg_width = int(fg_width * scale)
    fg_height = int(fg_height * scale)
    fg_image = fg_image.resize((fg_width, fg_height))

    bg_width, bg_height = ctx['image'].size

    done = False
    while not done:
        x0 = random.randint(0, bg_width - fg_width)
        y0 = random.randint(0, bg_height - fg_height)

        print(f'randomized {x0}, {y0}')

        box = {
            'x': x0 / bg_width,
            'y': y0 / bg_height,
            'width': fg_width / bg_width,
            'height': fg_height / bg_height,
        }

        done = not any([intersects(box, b) for b in ctx['boxes']])

    ctx['image'].paste(fg_image, (x0, y0), fg_image)
    ctx['boxes'].append(box)

    # Uncomment to draw bounding boxes round bees, useful when developing the script
    # draw = ImageDraw.Draw(ctx['image'])
    # draw.rectangle(((x0, y0), (x0 + fg_width, y0 + fg_height)), outline='blue', width=3)

    yolo_line = f'{DRONE_LABEL} {box["x"]} {box["y"]} {box["width"]} {box["height"]}'
    return yolo_line

if __name__ == '__main__':
    bg_file = random_file(BG_DIR)
    ctx = {
        'image': Image.open(bg_file),
        'boxes': [],
    }

    n_bees = random.randint(2,5)
    lines = [add_bee(ctx) for _ in range(n_bees)]

    out_image_file_name = Path(f'synthetic-{uuid.uuid4()}.jpg')
    out_annotation_file_name = out_image_file_name.with_suffix('.txt')

    ctx['image'].save(out_image_file_name, quality=95)
    with open(out_annotation_file_name, 'w') as f:
        f.write('\n'.join(lines))

    print(f'Generated {out_image_file_name} with {n_bees} bees and saved annotations in {out_annotation_file_name}')
    print(ctx['boxes'])
