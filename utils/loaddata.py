"""
Compatibility shim for old MAGIC Wget/StreamSpot pickles.

The original graphs.pkl was saved with classes from utils.loaddata.
We only need these class names so pickle can load the object.
"""

class StreamspotDataset:
    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]


class WgetDataset:
    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]
