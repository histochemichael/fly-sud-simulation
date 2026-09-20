# Individual behavior V4 — prospective protocol

The completed V3 pilot is immutable. This is a new behavioral calibration, not a retuned result.

## Design
Keep the consolidating memory mechanism and V3 time constants fixed to isolate controller changes. Test retention delays of 0.5 and 24 hours. These are fitted/calibration timing points, not held-out time-course evidence. Comparing and validating alternative memory mechanisms against independent timing data remains a separate unfinished scientific step.

Calibrate three absolute lateral-value gains (5, 20, 80), each with 2 reciprocal replicate pairs per delay (8 individual flies per delay). Common individual seeds across gain candidates allow controlled comparison; all validation seeds are disjoint.

Freeze one gain before validation: 8 reciprocal pairs per condition/delay, 4 flies per pair (both odor identities and both placements). Four conditions: paired, unpaired, untrained, retrieval DAN-silenced. Total validation: 256 independently simulated flies. Each individual is trained and tested once at its assigned delay. A pair is the inferential unit; this is substantially smaller per-group sampling than the paper.

## Individual variation
Independent seeds determine each fly's gait phases, initial heading, sensory noise, motor noise and learning gain. Learning gain is mean-one lognormal with log SD 0.3. Motor-noise SD is mean-0.18 lognormal with log SD 0.2, driving correlated zero-mean exploration with coefficient 0.9. Initial heading SD is 0.15 radians. These are explicit assumed distributions, not empirical estimates. No predetermined final choices, outcome balancing, artificial nonlearners, or post hoc jitter.

Learned steering uses tanh(gain times left-minus-right learned value), retaining memory magnitude. Innate, obstacle and upwind reflexes remain explicit assumed control components; anatomical connectivity does not specify these equations.

## Frozen selection and success rules
Selection loss per delay: (absolute CPI minus 0.35) squared, plus 4 for wrong sign, plus 4 times nonresponse fraction, plus 100 times physics failure fraction. Choose the lowest loss eligible candidate; break ties by lower gain. The 0.35 target is an engineering nonsaturation target, not a paper measurement.

Eligibility: negative early and positive late CPI; each magnitude 0.125 to 0.75; response at least 80%; zero physics failures. If none qualifies, stop and report calibration failure instead of silently searching.

Validation success: paired CPI negative early and positive late, magnitudes 0.05 to 0.8, both choices represented; untrained/unpaired absolute CPI at most 0.3; response at least 80% in each cell; no physics failures. Paired-minus-unpaired must have the expected sign and Holm-adjusted exact replicate sign-flip p below 0.05 at both delays. Report failures as failures. DAN retrieval comparison is exploratory.

CPI is the average of the two reciprocal odor-group preference indices, counting nonresponders in each denominator. Display mean CPI plus SEM across reciprocal pairs with replicate dots; separate individual choice/latency/path figures. Do not treat individual flies as independent CPI replicates.

## Provenance and limitations
Archive source, configs, dependency versions, source-data hashes, independent seeds, initial/final memory, full per-control sensory/motor state, and actual physical trajectories. Collection tips are absorbing: stop physical motion there, hold memory/assay until 120 seconds. No invented continuation.

Real FlyWire FAFB v783 selected connectivity remains active. The dopamine contribution is a simplified DAN-to-MBON retrieval gate, not a mechanistic DAN-to-KC plasticity simulation. This uses online associative learning, not deep learning or a whole-brain addiction model. No uploads.

Paper reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC4249630/

