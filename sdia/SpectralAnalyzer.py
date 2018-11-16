import numpy as np
import pandas as pd
import re
from scipy.sparse import csc_matrix
from tqdm import tqdm
from pyopenms import *
from .LibraryGenerator import SpectralLibrary
from sklearn.linear_model import Lasso


class SpectralAnalyzer:
    def __init__(self, lib_path, lib_name, ppm_tol=5., lam=1e8, is_silac=False):
        self.lib = SpectralLibrary.from_csv(lib_path + lib_name, is_silac)
        self.decoys = SpectralLibrary.from_csv(lib_path + lib_name + "_DECOY", is_silac)
        
        # Config
        self.ppm_tol = ppm_tol
        self.lam = lam
        self.is_silac = is_silac
        
    def extract_spectra(self, frame):
        mz, intensity = frame.get_peaks()
        mass_window = self.ppm_tol * mz * 1e-6 
        overlapping_peaks = np.searchsorted(mz + mass_window, mz - mass_window)
        unique_overlapping = np.unique(overlapping_peaks)
        if unique_overlapping.shape[0] != overlapping_peaks.shape[0]:
            mz = np.array(
                [np.mean(mz[np.where(overlapping_peaks == i)]) for i in unique_overlapping]
            )
            intensity = np.array(
                [np.sum(intensity[np.where(overlapping_peaks == i)]) for i in unique_overlapping]
            )
            
        return mz, intensity
    
    def bin_peaks(self, mz, library):
        mass_window = self.ppm_tol * mz * 1e-6
        intervals = np.sort(np.concatenate([mz - mass_window, mz + mass_window]))
        
        interval_positions = np.searchsorted(intervals, library.mz)
        frag_within_bound = np.mod(interval_positions + 1, 2) == 0
        
        library = library[frag_within_bound]
        bins = (interval_positions[frag_within_bound] - 1) // 2
        
        if self.is_silac:
            count_dict = dict(zip(*np.unique(library.isotope.values, return_counts=True)))
            fragment_select = np.array([count_dict[name] >= 3 for name in library.isotope.values])
        else:
            count_dict = dict(zip(*np.unique(library.index.values, return_counts=True)))
            fragment_select = np.array([count_dict[name] >= 3 for name in library.index.values])
            
        library = library[fragment_select]
        bins = bins[fragment_select]
            
        return library, bins
        
    def analyze(self, frame):
        # Extract spectra
        mz, intensity = self.extract_spectra(frame)
        
        # Filter libraries
        retention_range = frame.getRT() + 60 * 5. * np.array([-1., 1.])
        precursor = frame.getPrecursors()[0]
        mass_range = precursor.getMZ() + precursor.getIsolationWindowLowerOffset() * np.array([-1. , 1.])
        filtered_lib = self.lib.get_range(retention_range, mass_range)
        filtered_decoys = self.decoys.get_range(retention_range, mass_range)
        
        # Filter peaks
        filtered_lib, lib_bins = self.bin_peaks(mz, filtered_lib)
        filtered_decoys, decoy_bins = self.bin_peaks(mz, filtered_decoys)
        
        if filtered_lib.shape[0] == 0:
            return None, None
        
        # Build columns
        if self.is_silac:
            idx = filtered_lib.isotope.values
            decoy_idx = filtered_decoys.isotope.values
        else:
            idx = filtered_lib.index.values
            decoy_idx = filtered_decoys.index.values
            
        unique_idx = np.unique(idx)
        unique_decoy_idx = np.unique(decoy_idx)
        
        mapping_dict = {old: new for new, old in enumerate(unique_idx)}
        col_idx = np.array([mapping_dict[ind] for ind in idx], dtype=np.int64)
        
        decoy_mapping_dict = {old: new for new, old in enumerate(unique_decoy_idx)}
        offset = col_idx.max() + 1
        decoy_col_idx = np.array([decoy_mapping_dict[ind] + offset for ind in decoy_idx], dtype=np.int64)
        
        concat_cols = np.concatenate([col_idx, decoy_col_idx])
        
        # Build rows
        unique_bins = np.unique(np.concatenate([lib_bins, decoy_bins]))
        mapping_dict = {old: new for new, old in enumerate(unique_bins)}

        row_idx = np.array([mapping_dict[ind] for ind in lib_bins])
        decoy_row_idx = np.array([mapping_dict[ind] for ind in decoy_bins], dtype=np.int64)
        
        concat_rows = np.concatenate([row_idx, decoy_row_idx])
        
        # Build matrices
        sparse_library = csc_matrix((np.concatenate([filtered_lib.intensity.values.flatten(),
                                                     filtered_decoys.intensity.values.flatten()]), 
                                     (concat_rows, concat_cols)),
                                    shape=(concat_rows.max() + 1, concat_cols.max() + 1))
        regression_library = sparse_library.todense()
        
        # Do regression
        target = intensity[unique_bins]
        regressor = Lasso(alpha=self.lam, positive=True)
        regressor.fit(regression_library, target)

        selected_coef = regressor.coef_ > 0.
        peptide_select = selected_coef[:unique_idx.shape[0]]
        decoy_select = selected_coef[unique_idx.shape[0]:]

        peptide_return = unique_idx[peptide_select] if peptide_select.sum() > 0 else None
        decoy_return = unique_decoy_idx[decoy_select] if peptide_select.sum() > 0 else None

        return peptide_return, decoy_return
        
    def run(self, filename, name):
        file = MzMLFile()
        exp = MSExperiment()
        file.load(filename, exp)
        frame_list = []
        peptide_list = []
        decoy_list = []
        for frame_ind in tqdm(range(30017, 40001)):
            frame = exp[frame_ind]
            if frame.getMSLevel() == 2:
                frame_list.append(frame_ind)
                peptides, decoys = self.analyze(frame)
                peptide_list.append(peptides)
                decoy_list.append(decoys)

        with open(name + "_peptide_hits", "w") as dest:
            for idx in range(len(peptide_list)):
                dest.write(str(frame_list[idx]))
                if peptide_list[idx] is not None:
                    dest.write("," + ",".join(peptide_list[idx]))
                dest.write("\n")

        with open(name + "_decoy_hits", "w") as dest:
            for idx in range(len(decoy_list)):
                dest.write(str(frame_list[idx]))
                if decoy_list[idx] is not None:
                    dest.write("," + ",".join(decoy_list[idx]))
                dest.write("\n")
