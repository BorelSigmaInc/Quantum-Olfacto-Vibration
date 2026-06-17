# OCNQENC - Quantum Ocean Encoding

Amplitude-encoded global sea surface temperature (SST) states on a superconducting quantum processor.

## Authors

| Role | Name | ORCID |
|------|------|-------|
| Main Author | Raja Ram M | [0009-0008-4068-4079](https://orcid.org/0009-0008-4068-4079) |
| Main Author | Kryptur Quantum R&D | [0009-0009-2205-533X](https://orcid.org/0009-0009-2205-533X) |
| Contributor | Muskan S | [0009-0001-1968-0312](https://orcid.org/0009-0001-1968-0312) |
| Contributor | Vipul Jain | [0009-0009-9876-9312](https://orcid.org/0009-0009-9876-9312) |
| Contributor | Kalinga Swain | - |
| Data Manager | Borel Sigma Data Center | - |

**Affiliations:** Zius Quantum R&D Center, Data T Research Org, AE Quantum Research Division, Kryptur OU

**DOI:** [10.5281/zenodo.20736570](https://doi.org/10.5281/zenodo.20736570)

## Paper

Compile the report:

```bash
pdflatex Quantum_Ocean_Encoding.tex
pdflatex Quantum_Ocean_Encoding.tex
```

## Pipeline

1. `pipeline_step1_sst_to_circuit.py` - single-day SST to quantum circuit
2. `pipeline_multiday.py` - full January 2023 multi-day encoding
3. `ibm_fez_*.py` - hardware and simulation runs
4. `deep_visualization.py`, `panel_generator.py`, `card_generator.py` - figures

## Data

- `sst_data/` - NOAA OISST v2.1 daily NetCDF files (January 2023)
- `amplitude_vector.npz`, `encoding_circuit.qpy` - encoded quantum artefacts

## License

Research artefacts released for public fork and reuse. Cite DOI 10.5281/zenodo.20736570.
