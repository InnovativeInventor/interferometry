## Interferometry

This repo contains the analysis code and associated sample data for my interferometry lab in SRP. Rather than counting the interference patterns manually, I use fancy computer vision techniques (not really) to graph and count the number of interference patterns.

Hopefully this will aid and automate my collection of data, thereby increasing the accuracy of my results.

Note: This code is **very** messy. Don't write code like this. Seriously.

## Example usage
```
pipenv install
python3 interferometry.py
```

To calculate the number of peaks, run the Jupyter notebook like so:
```
ipython analysis.ipynb
```

Fine tuning can be done by editing `trials.json`.

## License
GPL-3.0-only
