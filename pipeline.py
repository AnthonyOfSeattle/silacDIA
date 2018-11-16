import numpy as np


reader = SptxtReader("./Data/spectral_library/merged/", "filtered_silac_cons")
generator = LibraryGenerator(reader, True, True, require_pairs=True)
slib = generator.build()
slib.to_csv("./Data/spectral_library/filtered_silac_library_b&y")

reader = SptxtReader("./Data/spectral_library/merged/", "filtered_silac_cons_DECOY")
generator = LibraryGenerator(reader, True, True, require_pairs=True)
slib = generator.build()
slib.to_csv("./Data/spectral_library/filtered_silac_library_b&y_DECOY")


lam_path = np.logspace(start=5, stop=10, num=11)

for lam_idx in range(lam_path.shape[0]):
    print("Solving for", lam_path[lam_idx])
    analyzer = SpectralAnalyzer(lib_path="./Data/spectral_library/",
                                lib_name="filtered_silac_library_b&y",
                                ppm_tol=5., lam=lam_path[lam_idx], is_silac=True)
    analyzer.run("./Data/DIA_FULL/UWPROFL0362.mzML", "./Results/UWPROFL0362" + "_" + str(lam_idx))