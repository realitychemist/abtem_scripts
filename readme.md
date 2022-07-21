# abTEM Scripts
This is a collection of personal-use scripts, not intended for public release.  Not all of them actually focus on abTEM, some of them focus on other aspects of TEM image/data processing.  There very well may be bugs in any/all of these scripts, so use anything you find here with due skepticism.

When a particular script is "finished" (i.e. I have used it to do what I need it to do), I'll document it here for my future reference:

## Finished Scripts
- `probe_wraparound_test` Used to check for probe wraparound for several likely convergence angles and sample thicknesses.  The ultimate conclusion was that an 18mrad probe works fine (in terms of wraparound) as long as the sample is less than about 15nm thick, and other convergence angles don't give much benefit here.  Future simulations will aim to keep models under 15nm as a result.  The full matrix of exit waves is shown in `_output/probetest.png`

