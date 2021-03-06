import numpy as np
import pandas as pd
import re
from tqdm import tqdm


class SpectralLibrary:
    def __init__(self, library, meta, is_silac=False):
        self.library = library
        self.meta = meta
        self.is_silac = is_silac

        # Create internal indexing
        peptide_dict = {name: ind for ind, name in enumerate(self.meta.index)}
        self.pep_idx = np.array([peptide_dict[name] for name in self.library.index])

        if is_silac:
            isotope_dict = {name: ind for ind, name in enumerate(self.meta.isotope)}
            self.isotope_idx = np.array([isotope_dict[name] for name in self.library.isotope])

    @classmethod
    def from_csv(cls, lib_name, is_silac=False):
        library = pd.read_csv(lib_name + "_SPECTRA.csv", index_col=0)
        meta = pd.read_csv(lib_name + "_META.csv", index_col=0)

        return cls(library, meta, is_silac)

    def to_csv(self, lib_name):
        self.library.to_csv(lib_name + "_SPECTRA.csv")
        self.meta.to_csv(lib_name + "_META.csv")

    def get_range(self, retention_range, mass_range, silac="strict"):
        retention_select = np.searchsorted(np.array(retention_range), self.meta.retention) == 1
        window_select = np.searchsorted(np.array(mass_range), self.meta.precursor) == 1
        select = np.logical_and(retention_select, window_select)

        if self.is_silac:
            return self.library[select[self.isotope_idx]]
        else:
            return self.library[select[self.pep_idx]]


class SptxtReader:
    def __init__(self, dir, basename):
        self.dir_ = dir
        self.basename_ = basename

        self.total_spectra = 0
        self.distinct_ions = 0
        self.stripped_peptides = 0
        self.read_pepidx_header_()

        self.file_handle_ = open(dir + basename + '.sptxt', 'r')

    def __del__(self):
        self.file_handle_.close()

    def __len__(self):
        return self.total_spectra

    def __iter__(self):
        return self

    def __next__(self):
        self.header = self.get_next_header_()

        if len(self.header) > 0:
            self.spectra = self.get_next_spectra_()

            return self.header, self.spectra

        else:
            raise StopIteration

    def read_pepidx_header_(self):
        with open(self.dir_ + self.basename_ + '.pepidx', 'r') as pepidx:
            for line in pepidx:
                if re.search('#', line) is None:
                    break

                elif re.search('spectra in library', line) is not None:
                    self.total_spectra = int(re.search('[0-9]+', line).group())

                elif re.search('distinct peptide_ions', line) is not None:
                    self.distinct_ions = int(re.search('[0-9]+', line).group())

                elif re.search('distinct stripped peptides', line) is not None:
                    self.stripped_peptides = int(re.search('[0-9]+', line).group())

    def get_next_header_(self):
        header = {}
        for line in self.file_handle_:
            line = line.split()
            if len(line) == 0 or line[0] == '###':
                continue

            else:
                if line[0] == 'Name:':
                    header['peptide'] = line[1]
                    header['charge'] = int(line[1][-1:])

                elif line[0] == 'MW:':
                    header['mw'] = float(line[1])

                elif line[0] == 'PrecursorMZ:':
                    header['precursor'] = float(line[1])

                elif line[0] == "Comment:":
                    # First pull out rt entry
                    for entry in line:
                        if re.search('Retention', entry) is not None:
                            header["retention"] = float(re.search('[0-9]+\.[0-9]+', entry).group())
                            break

                elif line[0] == 'NumPeaks:':
                    header['npeaks'] = int(line[1])
                    break

        return header

    def get_next_spectra_(self):
        npeaks = self.header['npeaks']
        mz = [np.NaN] * npeaks
        intensity = [np.NaN] * npeaks
        name = [np.NaN] * npeaks
        error = [np.NaN] * npeaks

        for p in range(npeaks):
            values = next(self.file_handle_).split()
            mz[p] = float(values[0])
            intensity[p] = float(values[1])

            peak_name = values[2]
            if peak_name == "?":
                name[p] = peak_name

            else:
                if peak_name[0] == "[":
                    peak_name = peak_name[1:-1]
                peak_name = peak_name.split(",")[0]  # Focusing on first annotation
                peak_name = peak_name.split("/")
                name[p] = peak_name[0]
                try:
                    error[p] = float(peak_name[1])
                except:
                    print(values)
                    raise

        spectra = pd.DataFrame({"mz": mz,
                                "intensity": intensity,
                                "name": name,
                                "error": error})
        return spectra


class LibraryGenerator:
    def __init__(self, reader, use_y=True, use_b=True, min_peaks=10, ppm=10, require_pairs=False):
        self.reader = reader
        self.min_peaks = min_peaks
        self.ppm = ppm
        self.require_pairs = require_pairs

        frag_dict = dict(y=use_y, b=use_b)
        fragments = '({})'.format('|'.join([f for f in frag_dict if frag_dict[f]]))
        self.fragment_re = re.compile(fragments + '([3-9]|([1-9]+[0-9]))' + '(-[0-9]+)*(\^2)*i*')

    def filter_(self, spectra):
        fragment_select = spectra.name.str.contains(self.fragment_re)
        spectra = spectra[fragment_select]

        ppm_select = spectra.error.abs() <= self.ppm * spectra.mz * 1e-6
        spectra = spectra[ppm_select]

        return spectra

    def silac_filter_(self, library, meta):
        print("Regex")
        search_value = re.compile('K[^a-zA-Z\/]+')
        name_dict = {p: re.sub(search_value, "K", p) for p in meta.index}

        print("Mutating Indicies")
        library.reset_index(inplace=True)
        library.rename(columns={"peptide": "isotope"}, inplace=True)
        library.index = pd.Index(library.isotope, name="peptide")
        library.rename(name_dict, inplace=True, level="peptide")

        meta.reset_index(inplace=True)
        meta.rename(columns={"peptide": "isotope"}, inplace=True)
        meta.index = pd.Index(meta.isotope, name="peptide")
        meta.rename(name_dict, inplace=True, level="peptide")

        print("Counting")
        peptide_counts = meta.groupby("peptide").size()
        final_peptides = peptide_counts[peptide_counts == 2].index.values

        return library.loc[final_peptides], meta.loc[final_peptides]

    def build(self):
        next(self.reader)
        passing = 0
        meta = []
        library = []
        for entry in tqdm(self.reader, "Generating sparse spectral matrix"):
            filtered_spec = self.filter_(entry[1])
            if filtered_spec.shape[0] >= self.min_peaks:
                filtered_spec.index = pd.Index(np.repeat(entry[0]["peptide"], filtered_spec.shape[0]),
                                               name="peptide")
                library.append(filtered_spec.copy())
                meta.append(pd.DataFrame(entry[0], index=[1]))
                passing += 1

                # if (passing + 1) % 1000 == 0:
                #     break

        print("Building full libraries")
        library = pd.concat(library)
        meta = pd.concat(meta).set_index("peptide")

        if self.require_pairs:
            library, meta = self.silac_filter_(library, meta)
            return SpectralLibrary(library, meta, is_silac=True)

        return SpectralLibrary(library, meta, is_silac=False)
