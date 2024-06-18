CELEBA_SENS_FEAS = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young".split(' ')
TOXIC_SENS_FEAS = [
    "toxicity",
    "male",
    "female",
    "transgender",
    "other_gender",
    "na_gender",  # 1-5；5
    "heterosexual",
    "homosexual_gay_or_lesbian",
    "bisexual",
    "other_sexual_orientation",
    "LGBTQ",
    "na_orientation",  # 6-11；6
    "christian",
    "jewish",
    "muslim",
    "hindu",
    "buddhist",
    "atheist",
    "other_religion",
    "other_religions",
    "na_religion",  # 12-20；9
    "black",
    "white",
    "asian",
    "latino",
    "other_race_or_ethnicity",
    "asian_latino_etc",
    "identity_any",
    "na_race",  # 21-28；8
    "physical_disability",
    "intellectual_or_learning_disability",
    "psychiatric_or_mental_illness",
    "other_disability",
    "disability_any",
    "na_disability",  # 29-34； 6
]
SENS_FEAS_ALL = {
                    "celebA": CELEBA_SENS_FEAS,
                    "toxic": TOXIC_SENS_FEAS,
                }   


CELEBA_2_FEAS_ID = [
                3,
                4,
                5,
                6,
                7,
                9,
                10,
                12,
                13,
                14,
                15,
                16,
                17,
                19,
                20,
                23,
                31,
                33,
                36,
                39,
            ]

CELEBA_39_FEAS_ID = [
                2,
                3,
                4,
                5,
                6,
                7,
                9,
                10,
                12,
                13,
                14,
                15,
                16,
                17,
                19,
                20,
                23,
                31,
                33,
                36,
            ]

TOXIC_25_FEAS_ID = [1,2,3,4,6,7,8,9,10,12,13,14,15,16,17,18,21,22,23,24,25,29,30,31,32]