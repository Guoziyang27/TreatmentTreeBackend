import os.path
import pandas as pd
from mimic.mimic_dataset import mimicDataset
from mimic.utils import *
from mimic.sample_name import *
from AI_Clinician_core import AI_Clinician_core, test_model

def main():
    # MIMICtable = mimicDataset()

    # AI_Clinician_core(MIMICtable)

    test_model()


if __name__ == '__main__':
    main()