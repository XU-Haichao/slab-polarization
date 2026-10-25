4U 1624-49 observational processing exports
==========================================

Entry point: ../../observational_analysis.ipynb
Source products: standard-v4-20261007; adopted fits: profile-reference-20261014.
The notebook displays cached best-fit values and predictions, never refits.

selection.json: actual ordinary-count-rate GTI choices and exposure partitions.
gti.json: adopted and alternative GTIs in recorded IXPE time coordinates.
lightcurve.npz: original 16 s ordinary source-count light curve, exposure and validity.
event_images.npz/json: calibrated pre-nondip event histograms, regions, source hashes.
ungrouped_spectra.json: native extraction channels, source/background rates/errors,
  exposures, BACKSCAL/AREASCAL, channel-energy bounds, original fitting grouping.
data-bindings.json: original spectra, backgrounds and response identities/hashes.
commands.json: original recorded extraction command arguments; not executed.
ixpe-event-manifest.json: event selection, exposure and deadtime records.
export_audit.json: data checksums and explicit no-refit/no-reextraction flags.

All paths use portable placeholders. ${OBS_ROOT} denotes the original source
analysis tree; ${EXTERNAL_RESOURCE} is an external calibration/response alias.
The actual response identity is retained in the logical path and SHA256.
These placeholders are documentary, not required notebook configuration.

We export original ungrouped PHA channel values, not unfolded fluxes. The images
are calibrated events (IXPE after historical particle rejection), not raw telemetry.
The native fit plots use original grouped fitting bins, without extra rebinning.
Q/U remain separate for each detector, with signed errors from the source files.

Large events, L1 inputs, original calibration trees, response matrices and
external model tables are not included. This is not a complete executable
telemetry-to-fit reduction. The notebook documents limits to available provenance.
No raw data or existing fit products were modified to produce these exports.
