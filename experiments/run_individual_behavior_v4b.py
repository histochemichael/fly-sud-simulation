"""Declared follow-up after the completed V4 gain bracket failed."""
import sys
import run_individual_behavior_v4 as base
base.POLICY={**base.POLICY,
 "version":"individual_behavior_v4b",
 "prior_information":"V4 gains 5 and 20 lacked early aversion; gain 80 saturated late preference. This is adaptive model development, not independent confirmation.",
 "candidate_gains":[50.],"calibration_seed":410724,"validation_seed":920618,
 "calibration_replicates":4,
 "calibration_selection":"Single intermediate gain 50; no candidate selection. Apply unchanged eligibility gate to 16 new flies per delay.",
 "followup_budget":"One follow-up calibration only. If it fails, stop and report; do not expand the search."}
if __name__=="__main__":
 if "--output" not in sys.argv:sys.argv+=["--output","results/individual_behavior_v4b"]
 base.main()

