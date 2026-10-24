"""Apply the established CACTI real-figure display orientation to exported PNGs.

This is a display transform after inference, never a sensing-mask transform.
Raw PNGs are retained in a separate directory. Hand is unchanged; Domino,
Pendulum Ball and Water Balloon use transpose, 45-degree counterclockwise
bicubic rotation without expansion, then a horizontal mirror.
"""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image


def orient(image, scene):
    if scene.lower() == 'hand':
        return image
    if scene.lower() not in ('duomino', 'domino', 'pendulumball', 'waterballoon'):
        raise ValueError(f'Unknown real-benchmark scene: {scene}')
    return (image.transpose(Image.Transpose.TRANSPOSE)
            .rotate(45, resample=Image.Resampling.BICUBIC, expand=False)
            .transpose(Image.Transpose.FLIP_LEFT_RIGHT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error('Use a separate output directory to preserve the raw frames')
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(args.input.glob('*.png')):
        scene = path.name.split('_')[0]
        with Image.open(path) as image:
            result = orient(image.convert('L'), scene)
            result.save(args.output/path.name)
        records.append({'file':path.name, 'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'output_sha256':hashlib.sha256((args.output/path.name).read_bytes()).hexdigest()})
    (args.output/'display_transform.json').write_text(json.dumps({
        'transform':__doc__, 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'frames':records}, indent=2)+'\n')
    print(f'Oriented {len(records)} frames into {args.output}')


if __name__ == '__main__': main()
