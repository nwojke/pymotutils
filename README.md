# pymotutils

A Python package that provides commonly used functionality when
implementing and testing algorithms for multiple object tracking.
This includes

* Preprocessing and postprocessing
* Datasets and evaluation toolkits
* Visualization

# Installation

The library can be installed with pip:

```
pip install git+https://github.com/nwojke/pymotutils
```

Note that some of the dataset wrappers in ``contrib`` may require additional
packages that are not listed as a dependency.

## Example

The following example downloads the MOT16 dataset and plays back one of
the training sequences:

```
wget https://motchallenge.net/data/MOT16.zip
unzip MOT16.zip -d MOT16
PYTHONPATH=$(pwd) python examples/motchallenge_playback.py \
    --mot_dir=./MOT16/train --sequence=MOT16-02
```

A complete implementation of a tracking method using this utility package
can be found in a seperate [project](https://github.com/nwojke/mcf-tracker).
