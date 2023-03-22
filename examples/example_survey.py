import ehtim as eh
import multiprocessing
from multiprocessing import set_start_method
from multiprocessing import Pool

#try:
#    set_start_method("spawn")
#except RuntimeError:
#    print("context already set")
def run_pset(pset, params_fixed):
    ps = eh.survey.ParameterSet(pset, params_fixed)
    ps.run()

if __name__ == '__main__':


    # Load in example image and array
    im = eh.image.load_txt('../models/avery_sgra_eofn.txt')
    eht = eh.array.load_txt('../arrays/EHT2017.txt')

    # create example observation uvfits file
    tint_sec = 5
    tadv_sec = 600
    tstart_hr = 0
    tstop_hr = 24
    bw_hz = 4e9
    obs = im.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz,
                     sgrscat=False, ampcal=True, phasecal=False)
    obs.save_uvfits('../../example_survey_obs.uvfits')

    sites = list(obs.tkey.keys())
    SEFD_errs = {}
    for site in sites:
        SEFD_errs[site] = 0.1

    print(SEFD_errs)

    # create a dict of non-varying imaging parameters, such as filenames
    # see a full list of fixed parameters in the docs
    params_fixed = eh.survey.create_params_fixed(infile='../../example_survey_obs.uvfits',
                                                 outfile_base='example_survey',
                                                 outpath='../../example_survey_output/',
                                                 nproc=-1,  # use all available cores, can be changed
                                                 overwrite=True,
                                                 niter_static=1,
                                                 SEFD_error_budget=SEFD_errs)

    psets = eh.survey.create_survey_psets(zbl=[0.6, 0.7], sys_noise=[0.01], prior_fwhm=[50])
    args = []
    print(len(psets))
    
    for i in range(2):
        args.append([dict(psets.iloc[i]), params_fixed])

    with multiprocessing.get_context('spawn').Pool() as pool:
#    with Pool() as pool:
        res = pool.starmap(run_pset, args)
