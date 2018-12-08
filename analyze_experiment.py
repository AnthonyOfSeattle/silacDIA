from sdia import *
import sys
import numpy as np

experiment = sys.argv[1]
str_pen = sys.argv[2]
pen = float(str_pen)
str_pen_2 = sys.argv[3]
pen_2 = float(str_pen_2)

# Lasso experiment
print("Analyzing using LASSO")
analyzer = SpectralAnalyzer(lib_path="/net/gs/vol1/home/valenta4/silacDIA/Data/spectral_library/",
                            lib_name="filtered_silac_library_b&y",
                            ppm_tol=5., lam=pen, is_silac=True, model="lasso")
analyzer.run("/net/gs/vol1/home/valenta4/silacDIA/Data/DIA_FULL/" + experiment + ".mzML", 
             "/net/gs/vol1/home/valenta4/silacDIA/Results/" + experiment +  "_" + "lasso" + "_" + str(str_pen))

# Group lasso experiment
print("Analyzing using Group LASSO")
analyzer = SpectralAnalyzer(lib_path="/net/gs/vol1/home/valenta4/silacDIA/Data/spectral_library/",
                            lib_name="filtered_silac_library_b&y",
                            ppm_tol=5., lam=pen, is_silac=True, model="glasso")
analyzer.run("/net/gs/vol1/home/valenta4/silacDIA/Data/DIA_FULL/" + experiment + ".mzML", 
             "/net/gs/vol1/home/valenta4/silacDIA/Results/" + experiment + "_" + "glasso" + "_" + str(str_pen))

# Sparse Group lasso experiment
print("Analyzing using Sparse Group LASSO")
analyzer = SpectralAnalyzer(lib_path="/net/gs/vol1/home/valenta4/silacDIA/Data/spectral_library/",
                            lib_name="filtered_silac_library_b&y",
                            ppm_tol=5., lam=pen, lam2=pen_2, is_silac=True, model="sglasso")
analyzer.run("/net/gs/vol1/home/valenta4/silacDIA/Data/DIA_FULL/" + experiment + ".mzML", 
             "/net/gs/vol1/home/valenta4/silacDIA/Results/" + experiment + "_" + "sglasso" + "_" + str_pen + "_" + str_pen_2)