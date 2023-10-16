import os
from os import system
from subprocess import Popen
from omc3.scripts.fake_measurement_from_model import generate as fake_measurement
from omc3 import global_correction
from omc3.response_creator import create_response_entrypoint
from sortq2withuncertainty import Pairs, Q2Pairs
from sortq1withuncertainty import Q1Pairs
from magnet_errors import *
from pathlib import Path
import tfs
from matplotlib import pyplot as plt, sys
import numpy as np
import pandas as pd
from typing import List, Union, Tuple

MADX = "/home/awegsche/programs/madx/madx-gnu64"
MODEL_TWISS = "model/twiss_err_b1.tfs"
EFILE =  "./table_corrections_Q2.madx"
SEED = 0
VAR_CATS = [
        'kqx2a.l1', 'kqx2a.r1',
        'kqx2a.l5', 'kqx2a.r5',
        'kqx2b.l1', 'kqx2b.r1',
        'kqx2b.l5', 'kqx2b.r5',
        'kqx1.l1', 'kqx1.r1',
        'kqx1.l5', 'kqx1.r5',
        #'ktqx1.l1', 'ktqx1.r1',
        #'MQY',
        ]


DO_MADX = True
DO_FR = True
DO_CORR = True

AMP_REAL_ERROR = 10
AMP_MEAS_ERROR = 2

SUMM_PAIRING = "summ_pairing.tfs"
SUMM_SUM = "summ_sum.tfs"
SUMM_DIFF = "summ_diff.tfs"


def write_summary(parameters, filename):
    print("writing summary tfs")
    tfs_summary = tfs.read(filename) if os.path.exists(filename) else pd.DataFrame()
    index = len(tfs_summary.index)

    tfs.write(filename,
              pd.concat([tfs_summary, pd.DataFrame(parameters, index=[index])])
              )

def main():
    """
    """

    for i in range(1000):
        do_analysis()


def do_analysis():

    q2_errors = Q2Pairs(10,2)
    q1_errors = Q1Pairs(10,2)

    try:
        check1, err1, diff1 = do_sim(q1_errors, q2_errors)
    except:
        return

    def sort_and_sim(score_fun, summ_filename):
        """ applies the sorting, then simulates the sorted lattice, measures beta beating before
        and after corrections
        """
        # ---- Q1 / Q3 errors ----------------------------------------------------------------------
        print(f"searching for best combination in {len(q1_errors.permutations)} permutations")
        print(f"using the diff method")

        score = score_diff(q1_errors)
        print(f" -- initial score: {score}")

        next_progress = 0.1
        best_comb = 0
        
        for i in range(len(q1_errors.permutations)):
            if i / len(q1_errors.permutations) > next_progress:
                print(f"progress: {100* i / len(q1_errors.permutations):.0f} %")
                next_progress += 0.1

            q1_errors.selected_permutation = i

            sum = score_diff(q1_errors)

            if sum < score:
                score = sum
                best_comb = i
                
        q1_errors.selected_permutation = best_comb
        print(f"final score: {score}")

        # ---- Q2 errors ---------------------------------------------------------------------------
        print(f"searching for best combination in {len(q2_errors.permutations)} permutations")
        print(f"using the diff method")

        score = score_fun(q2_errors)
        print(f" -- initial score: {score}")

        next_progress = 0.1
        
        for i in range(len(q2_errors.permutations)):
            if i / len(q2_errors.permutations) > next_progress:
                print(f"progress: {100* i / len(q2_errors.permutations):.0f} %")
                next_progress += 0.1

            q2_errors.selected_permutation = i

            sum = score_fun(q2_errors)

            if sum < score:
                score = sum
                best_comb = i
                
        print(f"final score: {score}")

        q2_errors.selected_permutation = best_comb

        # ---- so simulations, write results to df ------------------------------------------------
        check2, err2, diff2 = do_sim(q1_errors, q2_errors)

        params = {}
        q2_errors.log_strengths(params)
        q1_errors.log_strengths(params)
        params["BBEAT"] = err1
        params["BBEAT_AFTER"] = err2
        params["CORR"] = diff1
        params["CORR_AFTER"] = diff2
        params["CHECK"] = check1
        params["CHECK_AFTER"] = check2

        write_summary(params, summ_filename)

        # ---- end of sort_and_sim ----------------------------------------------------------------


    def score_diff(errors: Pairs):
        return np.sum([
            (errors.get_pair(i)[0].real_error - errors.get_pair(i)[1].real_error)**2
            for i in range(errors.get_pair_count())])

    def score_sum(errors):
        return np.sum([
            (errors.get_pair(i)[0].real_error + errors.get_pair(i)[1].real_error)**2
            for i in range(errors.get_pair_count())])

    try:
        sort_and_sim(score_diff, SUMM_DIFF)
        sort_and_sim(score_sum, SUMM_SUM)
    except:
        return



def do_sim(q1_errors: Q1Pairs, q2_errors: Q2Pairs) -> Tuple[float, float, float]:
    print("running simulation")

    if DO_MADX:

        print("running madx")

        q2_errors.write_errors_to_file("errors_Q2.madx")
        q1_errors.write_errors_to_file("errors_Q1.madx")

        template_file = open("job.inj.madx", "r")
        jobfile = open("run_job.inj.madx", "w")
        jobfile.write(template_file.read().replace("_TRACK_", "0")) # no tracking
        template_file.close()
        jobfile.close()

        # remove twiss output, its existance after running madx indicates success
        system(f"rm {MODEL_TWISS}")

        with open(os.devnull, "w") as devnull:
            Popen([MADX, "run_job.inj.madx"], stdout=devnull, cwd="model").wait()

        if not os.path.exists(f"{MODEL_TWISS}"):
            raise RuntimeError("twiss_err_b1.tfs does not exist")
            

    accel_params = dict(
        accel="lhc",
        year="hl16", # to be checked
        beam=1,
        model_dir=Path("model1").absolute(),
            )


    if DO_FR:
        print("creating response entrypoint")
        create_response_entrypoint(
            outfile_path=Path("/home/awegsche/fellow/40_magnet_sorting/model1/FullResponse.h5"),
            creator="madx",
            optics_params=['PHASEX', 'PHASEY', 'BETX', 'BETY', 'Q'],
            variable_categories=VAR_CATS,
            delta_k=1.0e-7,
            **accel_params 
        )


    # fake measurement
    fake_measurement(twiss=MODEL_TWISS,
                     model="model1/twiss.dat",
                     outputdir="fake_measurements")


    if DO_CORR:
        print("running global correction")
        global_correction.global_correction_entrypoint(
            meas_dir="fake_measurements",
            output_dir="global_corrections",
            fullresponse_path="model1/FullResponse.h5",
            iterations=1,
            optics_params=['PHASEX', 'PHASEY', 'BETX', 'BETY', 'Q'],
            variable_categories=VAR_CATS,
            **accel_params,
                )
        system("cp scripts_global_corrs/rerun.job.madx global_corrections/rerun.job.madx")
        with open(os.devnull, "w") as devnull:
            Popen([MADX, "rerun.job.madx"], cwd="global_corrections", stdout=devnull).wait()




    # compare
    check_twiss = tfs.read("global_corrections/twiss_global_b1.tfs")
    err_twiss = tfs.read(MODEL_TWISS)
    model_twiss = tfs.read("model1/twiss.dat")

    #plt.plot(check_twiss["S"], check_twiss["BETX"]/model_twiss["BETX"]-1, label="check")
    #plt.plot(err_twiss["S"], err_twiss["BETX"]/model_twiss["BETX"]-1, label="err")
    #plt.legend()

    #plt.show()

    rms_check = rms(check_twiss["BETX"]/model_twiss["BETX"]-1)
    rms_err = rms(err_twiss["BETX"]/model_twiss["BETX"]-1)
    rms_diff = rms(check_twiss["BETX"]/err_twiss["BETX"]-1)

    print(f"rms check: {rms_check}, rms err: {rms_err}")
    print(f"rms diff: {rms_diff}")

    return rms_check, rms_err, rms_diff


def rms(array):
    return np.sqrt(np.mean(array**2))

main()