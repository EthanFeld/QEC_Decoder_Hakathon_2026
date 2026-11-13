# ducks are pretty cool — adaptive decoder

Team: **ducks are pretty cool**. Kaggle name: **Ethan Feldman**.
Selected adaptive submission: **58322820**, public score **0.03370**.

This is the last validated CPU hand-in, with an offline asset bootstrap added
for a source-only GitHub checkout. It supports the two included public circuits.
No GPU implementation or GPU performance claim is included in this version.

From this directory, install the pinned dependencies before going offline:

```sh
python -m pip install -r requirements.txt
```

Then use the organizer entry:

```python
import decoder  # Finish initialization before starting the inference timer.
predictions = decoder.decode(circuit, detector_events)  # uint8[shots, 4]
probabilities = decoder.predict_proba(circuit, detector_events)
metadata = decoder.get_startup_metadata()
decoder.close()
```

The target is the organizer's 288 GB RAM, 48-core host. Allow at least roughly
60 GiB available RAM for the automatic adaptive startup; its explicit memory
and Linux cgroup checks determine the exact requirement. Initialization builds
missing boundary tables and the two 10 GiB physical row banks from the public
circuit model, prepares bounded endpoint caches, and warms native Numba kernels.
This can take substantial time and happens before inference timing.
`NUMBA_NUM_THREADS=48` requests 48 native CPU workers, capped at 48; BLAS is
limited to one worker before this entry imports numerical dependencies.

The two boundary `.npz` files exceed GitHub's file-size limit and are omitted.
`decoder` automatically rebuilds missing files into `data/target03/` using the
included `diagnostics.boundary21_tables` arithmetic. To perform that step first:

```sh
python setup_assets.py --threads 48
```

This builder uses no labels, saved predictions, training data, or network
services. It creates only circuit-derived probability tables. Original asset
checksums are recorded in `publication_manifest.json`; rebuilt file bytes can
differ because their metadata records the packaged source hashes. Existing
assets are left in place, with schema/circuit validation performed by the
runtime loader. Small machines may use separately built physical row stores
with `QEC_ROW_STORAGE=disk`; those stores are not included in this repository.

The decoding policy is the validated retained probability model plus up to
16 selected omitted rows per edge when a retained marginal lies in [0.4, 0.6].
It makes four independent strict `q > 0.5` decisions. Caches contain circuit
physics, never complete syndrome answers. Unknown circuits raise an error.

`VALIDATED_ARCHIVE_MANIFEST.json` preserves the original CPU ZIP inventory.
`publication_manifest.json` lists the source-only publication files and their
hashes. The newly added bootstrap was reviewed statically; a full rebuild was
not run during publication preparation. No credentials, competition inputs,
logical labels, prediction CSVs, or compiled caches are published.
