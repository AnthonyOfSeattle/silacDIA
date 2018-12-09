from sdia import *
import numpy as np


reader = SptxtReader("./Data/spectral_library/merged/", "filtered_silac_cons")
generator = LibraryGenerator(reader, True, True, require_pairs=True)
slib = generator.build()
slib.to_csv("./Data/spectral_library/filtered_silac_library_b&y")

reader = SptxtReader("./Data/spectral_library/merged/", "filtered_silac_cons_DECOY")
generator = LibraryGenerator(reader, True, True, require_pairs=True)
slib = generator.build()
slib.to_csv("./Data/spectral_library/filtered_silac_library_b&y_DECOY")