# pymotutils

A Python package that provides commonly used functionality when
implementing and testing algorithms for multiple object tracking.
This includes

* Preprocessing and postprocessing
* Datasets and evaluation toolkits
* Visualization

## Dependencies

* NumPy
* sklearn (linear assignment)
* OpenCV (visualization)

## Example

The following example downloads the MOT16 dataset and plays back one of
the training sequences: 

```
wget https://motchallenge.net/data/MOT16.zip
unzip MOT16.zip -d MOT16
PYTHONPATH=$(pwd) python examples/motchallenge_playback.py \
    --mot_dir=./MOT16/train --sequence=MOT16-02
```
