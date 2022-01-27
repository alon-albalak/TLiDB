from distutils.util import change_root


ross = "Ross Geller"
monica = "Monica Geller"
chandler = "Chandler Bing"
phoebe = "Phoebe Buffay"
joey = "Joey Tribbiani"
rachel = "Rachel Green"

manual_mapping = {
    "train_13": {"dialogue_id": "s09_e09_c07", "turns": [0,0],
                "speaker_map":{
                    "Speaker 1": "Mike Hannigan", "Speaker 2": ross
                    }},
    "train_40": {"dialogue_id": "s06_e24_c11", "turns": [10,33],
                "speaker_map":{
                    "Speaker 1": joey, "Speaker 2": "Mr. Bowmont", "Speaker 3": rachel
                    }},
    "train_45": {"dialogue_id": "s06_e06_c06", "turns": [0,6],
                "speaker_map":{
                    "Speaker 1": monica, "Speaker 2": rachel,"Speaker 3": ross
                    }},
    "train_66": {"dialogue_id": "s01_e01_c02", "turns": [25,32],
               "speaker_map":{
                  "Speaker 1": chandler, "Speaker 2": "Paul the Wine Guy", "Speaker 3": monica,
                  "Speaker 4": joey, "Speaker 5": ross
                 }},
    "train_73": {"dialogue_id": "s01_e16_c10", "turns": [13,20],
                "speaker_map":{
                    "Speaker 1": monica, "Speaker 2": "Mr. Heckles","Speaker 3": rachel
                    }},
    "train_86": {"dialogue_id": "s09_e19_c11", "turns": [0,18],
                "speaker_map":{
                    "Speaker 1": phoebe, "Speaker 2": monica
                    }},
    "train_94": {"dialogue_id": "s01_e07_c16", "turns": [0,15],
                "speaker_map": {
                    "Speaker 1": ross, "Speaker 2": rachel,"Speaker 3": monica, "Speaker 4": monica, "Speaker 5": "Paolo"
                    }},
    "train_96": {"dialogue_id": "s08_e08_c07", "turns": [0,16],
                "speaker_map": {
                    "Speaker 1": "Leonard Green", "Speaker 2": ross,"Speaker 3": "Mona"
                    }},
    "train_132": {"dialogue_id": "s07_e03_c05", "turns": [19,20],
                "speaker_map": {
                    "Speaker 1": ross, "Speaker 2": "Jack Geller"
                    }},
    "train_206": {"dialogue_id": "s09_e08_c01", "turns": [250,253],
                "speaker_map": {
                    "Speaker 1": rachel, "Speaker 2": "Amy Green" 
                }},
    "train_244": {"dialogue_id": "s09_e08_c01", "turns": [109,112],
                "speaker_map": {
                    "Speaker 1": joey, "Speaker 2": phoebe
                }},
    "train_252": {"dialogue_id": "s10_e02_c08", "turns": [0,23],
                "speaker_map": {
                    "Speaker 1": phoebe, "Speaker 2": "Frank Buffay Jr."
                }},
    "train_308": {"dialogue_id": "s07_e10_c11", "turns": [19,28],
                "speaker_map": {
                    "Speaker 1": chandler, "Speaker 2": "Ben Geller", "Speaker 3": ross, "Speaker 4": monica
                }},
    "train_320": {"dialogue_id": "s10_e13_c01", "turns": [0,6],
                "speaker_map": {
                    "Speaker 1": phoebe, "Speaker 2": "All", "Speaker 3": rachel, "Speaker 4": chandler
                }},
    "train_336": {"dialogue_id": "s09_e20_c08", "turns": [0,12],
                "speaker_map": {
                    "Speaker 1": monica, "Speaker 2": rachel, "Speaker 3": "Matthew Ashford"
                }},
    "train_343": {"dialogue_id": "s07_e21_c04", "turns": [21,28],
                "speaker_map": {
                    "Speaker 1": chandler, "Speaker 2": joey, "Speaker 3": monica
                }},
    "train_371": {"dialogue_id": "s08_e06_c03", "turns": [151,173],
                "speaker_map": {
                    "Speaker 1": chandler, "Speaker 2": ross, "Speaker 3": joey, "Speaker 4": monica, "Speaker 5": "Mona"
                }},
    "train_382": {"dialogue_id": "s09_e03_c03", "turns": [0,15],
                "speaker_map": {
                    "Speaker 1": chandler, "Speaker 2": monica
                }},
    "train_420": {"dialogue_id": "s01_e09_c03", "turns": [0,6],
                "speaker_map": {
                    "Speaker 1": ross, "Speaker 2": "Susan Bunch"
                }},
    "train_428": {"dialogue_id": "s02_e09_c03", "turns": [0,23],
                "speaker_map": {
                    "Speaker 1": "Grandmother", "Speaker 2": phoebe
                }},
    "train_488": {"dialogue_id": "s09_e17_c12", "turns": [29,44],
                "speaker_map": {
                    "Speaker 1": "Kori", "Speaker 2": chandler, "Speaker 3": monica, "Speaker 4": ross
                }},
    "train_515": {"dialogue_id": "s01_e11_c09", "turns": [18,22],
                "speaker_map": {
                    "Speaker 1": rachel, "Speaker 2": phoebe
                }},
    "train_531": {"dialogue_id": "s03_e10_c10", "turns": [8,17],
                "speaker_map": {
                    "Speaker 1": "Leader", "Speaker 2": ross, "Speaker 3": chandler
                }},
    "train_539": {"dialogue_id": "s09_e20_c10", "turns": [0,8],
                "speaker_map": {
                    "Speaker 1": joey, "Speaker 2": ross, "Speaker 3": "Dirk"
                }},
    "train_557": {"dialogue_id": "s06_e13_c07", "turns": [0,6],
                "speaker_map": {
                    "Speaker 1": monica, "Speaker 2": chandler
                }},
    "train_562": {"dialogue_id": "s02_e14_c09", "turns": [7,14],
                "speaker_map": {
                    "Speaker 1": joey, "Speaker 2": chandler
                }},
    "train_571": {"dialogue_id": "s04_e24_c19", "turns": [0,7],
                "speaker_map": {
                    "Speaker 1": "Emily Waltham", "Speaker 2": joey, "Speaker 3": ross, "Speaker 4": "Jack Geller"
                }},
    "train_574": {"dialogue_id": "s02_e03_c03", "turns": [0,6],
                "speaker_map": {
                    "Speaker 1": "Mr. Treeger", "Speaker 2": monica, "Speaker 3": "Buddy Boyles", "Speaker 4": rachel
                }},
    "train_593": {"dialogue_id": "s07_e10_c01", "turns": [0,3],
                "speaker_map": {
                    "Speaker 1": chandler, "Speaker 2": monica
                }},
    "train_602": {"dialogue_id": "s08_e06_c05", "turns": [0,20],
                "speaker_map": {
                    "Speaker 1": chandler, "Speaker 2": ross, "Speaker 3": phoebe, "Speaker 4": "Ursula Buffay",
                    "Speaker 5": "Eric", "Speaker 6": joey
                }},
    "train_645": {"dialogue_id": "s09_e08_c01", "turns": [21,49],
                "speaker_map": {
                    "Speaker 1": rachel, "Speaker 2": ross, "Speaker 3": "Woman At Door", "Speaker 4": "Amy Green"
                }},
    "train_649": {"dialogue_id": "s10_e10_c05", "turns": [147,154],
                "speaker_map": {
                    "Speaker 1": chandler, "Speaker 2": monica
                }},
    "train_705": {"dialogue_id": "s02_e09_c12", "turns": [0,13],
                "speaker_map": {
                    "Speaker 1": joey, "Speaker 2": rachel, "Speaker 3": chandler, "Speaker 4": phoebe, "Speaker 5": ross
                }},
    "train_723": {"dialogue_id": "s10_e16_c04", "turns": [0,7],
                "speaker_map": {
                    "Speaker 1": phoebe, "Speaker 2": monica, "Speaker 3": ross
                }},
    "train_726": {"dialogue_id": "s10_e16_c07", "turns": [0,8],
                "speaker_map": {
                    "Speaker 1": rachel, "Speaker 2": monica
                }},
    "train_765": {"dialogue_id": "s09_e19_c08", "turns": [0,17],
                "speaker_map": {
                    "Speaker 1": ross, "Speaker 2": chandler, "Speaker 3": "Chandlers"
                }},
    "train_769": {"dialogue_id": "s01_e08_c03", "turns": [0,7],
                "speaker_map": {
                    "Speaker 1": monica, "Speaker 2": ross, "Speaker 3": "Aunt Lillian", "Speaker 4": "Judy Geller"
                }},
    "train_778": {"dialogue_id": "s02_e24_c13", "turns": [0,2],
                "speaker_map": {
                    "Speaker 1": ross, "Speaker 2": joey
                }},
    "train_797": {"dialogue_id": "s05_e15_c01", "turns": [8,22],
                "speaker_map": {
                    "Speaker 1": rachel, "Speaker 2": chandler, "Speaker 3": joey, "Speaker 4": ross, "Speaker 5": monica
                }},
    "train_835": {"dialogue_id": "s09_e16_c02", "turns": [0,17],
                "speaker_map": {
                    "Speaker 1": "Mike Hannigan", "Speaker 2": phoebe
                }},
    "train_836": {"dialogue_id": "s09_e11_c02", "turns": [0,14],
                "speaker_map": {
                    "Speaker 1": joey, "Speaker 2": phoebe
                }},
    "train_841": {"dialogue_id": "s06_e08_c08", "turns": [0,9],
                "speaker_map": {
                    "Speaker 1": monica, "Speaker 2": chandler
                }},
    "train_849": {"dialogue_id": "s04_e24_c23", "turns": [6,14],
                "speaker_map": {
                    "Speaker 1": joey, "Speaker 2": chandler
                }},
    "train_854": {"dialogue_id": "s09_e08_c01", "turns": [50,55],
                "speaker_map": {
                    "Speaker 1": monica, "Speaker 2": chandler
                }},
    "train_857": {"dialogue_id": "s07_e21_c03", "turns": [81,90],
                "speaker_map": {
                    "Speaker 1": monica, "Speaker 2": chandler, "Speaker 3": joey
                }},
    "train_861": {"dialogue_id": "s08_e12_c05", "turns": [0,6],
                "speaker_map": {
                    "Speaker 1": ross, "Speaker 2": "Student"
                }},
    "train_868": {"dialogue_id": "s04_e12_c07", "turns": [11,53],
                "speaker_map": {
                    "Speaker 1": ross, "Speaker 2": joey, "Speaker 3": chandler, "Speaker 4": rachel, "Speaker 5": monica
                }},
    "train_874": {"dialogue_id": "s04_e24_c18", "turns": [22,34],
                "speaker_map": {
                    "Speaker 1": "Minister", "Speaker 2": "Emily Waltham", "Speaker 3": "Minister", "Speaker 4": ross
                }},
    "train_897": {"dialogue_id": "s08_e21_c14", "turns": [41,44],
                "speaker_map": {
                    "Speaker 1": ross, "Speaker 2": rachel
                }},
    "train_908": {"dialogue_id": "s03_e09_c01", "turns": [0,7],
                "speaker_map": {
                    "Speaker 1": "The Guys", "Speaker 2": phoebe, "Speaker 3": monica, "Speaker 4": rachel
                }},
    "train_1036": {"dialogue_id": "s06_e18_c16", "turns": [0,8],
                "speaker_map": {
                    "Speaker 1": rachel, "Speaker 2": joey
                }},
    "train_1046": {"dialogue_id": "s09_e03_c15", "turns": [0,22],
                "speaker_map": {
                    "Speaker 1": ross, "Speaker 2": monica, "Speaker 3": chandler,
                    "Speaker 4": rachel, "Speaker 5": phoebe, "Speaker 6": joey
                }},
    "train_1047": {"dialogue_id": "s04_e24_c20", "turns": [16,23],
                "speaker_map": {
                    "Speaker 1": rachel, "Speaker 2": chandler, "Speaker 3": monica, "Speaker 4": phoebe
                }},
    "dev_40": {"dialogue_id": "s09_e09_c10", "turns": [0,0],
                "speaker_map": {
                    "Speaker 1": rachel, "Speaker 2": phoebe, "Speaker 3": ross, "Speaker 4": "Mike Hannigan"
                }},
    "dev_56": {"dialogue_id": "s05_e15_c01", "turns": [0,7],
                "speaker_map": {
                    "Speaker 1": ross, "Speaker 2": chandler, "Speaker 3": monica
                }},
    "dev_104": {"dialogue_id": "s04_e24_c20", "turns": [35,37],
                "speaker_map": {
                    "Speaker 1": chandler, "Speaker 2": monica, "Speaker 3": joey
                }},
    "dev_159": {"dialogue_id": "s09_e20_c10", "turns": [8,10],
                "speaker_map": {
                    "Speaker 1": monica, "Speaker 2": joey
                }},
    "dev_182": {"dialogue_id": "s02_e18_c03", "turns": [0,5],
                "speaker_map": {
                    "Speaker 1": chandler, "Speaker 2": "Eddie Menuek"
                }},
    "dev_217": {"dialogue_id": "s05_e11_c04", "turns": [11,14],
                "speaker_map": {
                    "Speaker 1": ross, "Speaker 2": "Ben Geller", "Speaker 3": monica
                }},
    "dev_229": {"dialogue_id": "s09_e05_c03", "turns": [25,28],
                "speaker_map": {
                    "Speaker 1": monica, "Speaker 2": chandler
                }},
    "dev_232": {"dialogue_id": "s10_e09_c13", "turns": [0,32],
                "speaker_map": {
                    "Speaker 1": "Erica", "Speaker 2": "Adoption Agency Guy", "Speaker 3": chandler,
                    "Speaker 4": "Agency Guy", "Speaker 5": monica
                }},
    "dev_244": {"dialogue_id": "s02_e17_c02", "turns": [3,12],
                "speaker_map": {
                    "Speaker 1": phoebe, "Speaker 2": chandler, "Speaker 3": "ALL",
                }},
    "dev_249": {"dialogue_id": "s05_e08_c14", "turns": [0,9],
                "speaker_map": {
                    "Speaker 1": chandler, "Speaker 2": monica
                }},
    "dev_280": {"dialogue_id": "s05_e11_c06", "turns": [18,21],
                "speaker_map": {
                    "Speaker 1": monica, "Speaker 2": chandler, "Speaker 3": rachel
                }},
    "dev_299": {"dialogue_id": "s02_e14_c10", "turns": [70,79],
                "speaker_map": {
                    "Speaker 1": ross, "Speaker 2": "Judy Geller", "Speaker 3": ross,
                    "Speaker 4": "Chip Matthews", "Speaker 5": monica, "Speaker 6": rachel, "Speaker 7": "Roy"
                }},
    "dev_302": {"dialogue_id": "s07_e01_c08", "turns": [0,19],
                "speaker_map": {
                    "Speaker 1": monica, "Speaker 2": ross, "Speaker 3": rachel, "Speaker 4": phoebe
                }},
    "dev_328": {"dialogue_id": "s07_e02_c08", "turns": [0,14],
                "speaker_map": {
                    "Speaker 1": monica, "Speaker 2": "Judy Geller", "Speaker 3": "Jack Geller", "Speaker 4": chandler
                }},
    "dev_351": {"dialogue_id": "s09_e20_c10", "turns": [14,24],
                "speaker_map": {
                    "Speaker 1": joey, "Speaker 2": rachel
                }},
    "test_14": {"dialogue_id": "s04_e24_c22", "turns": [14,19],
                "speaker_map": {
                    "Speaker 1": rachel, "Speaker 2": "Stephen Waltham", "Speaker 3": ross, "Speaker 4": "Andrea Waltham"
                }},
    "test_17": {"dialogue_id": "s06_e21_c07", "turns": [9,14],
                "speaker_map": {
                    "Speaker 1": joey, "Speaker 2": "Estelle Leonard"
                }},
    "test_91": {"dialogue_id": "s10_e13_c04", "turns": [16,25],
                "speaker_map": {
                    "Speaker 1": ross, "Speaker 2": "Leonard Green"
                }},
    "test_101": {"dialogue_id": "s04_e04_c08", "turns": [0,14],
                "speaker_map": {
                    "Speaker 1": "Mr. Treeger", "Speaker 2": joey
                }},
    "test_118": {"dialogue_id": "s09_e10_c08", "turns": [0,35],
                "speaker_map": {
                    "Speaker 1": chandler, "Speaker 3": monica,
                    "Speaker 4": "Others", "Speaker 5": phoebe, "Speaker 6": ross,
                    "Speaker 7": joey, "Speaker 8": rachel
                }},
    "test_121": {"dialogue_id": "s09_e08_c01", "turns": [136,153],
                "speaker_map": {
                    "Speaker 1": joey, "Speaker 2": monica, "Speaker 3": "Amy Green",
                    "Speaker 4": rachel, "Speaker 5": chandler, "Speaker 6": ross
                }},
    "test_146": {"dialogue_id": "s09_e19_c12", "turns": [11,26],
                "speaker_map": {
                    "Speaker 1": rachel, "Speaker 2": joey
                }},
    "test_156": {"dialogue_id": "s06_e11_c07", "turns": [0,22],
                "speaker_map": {
                    "Speaker 2": joey, "Speaker 3": chandler, "Speaker 4": monica
                }},
    "test_192": {"dialogue_id": "s09_e08_c01", "turns": [94,108],
                "speaker_map": {
                    "Speaker 1": rachel, "Speaker 2": "ALL", "Speaker 3": "Amy Green",
                    "Speaker 4": joey, "Speaker 5": ross, "Speaker 6": monica, "Speaker 7": phoebe
                }},
    "test_214": {"dialogue_id": "s09_e08_c01", "turns": [10,20],
                "speaker_map": {
                    "Speaker 1": joey, "Speaker 2": "Tv Announcer", "Speaker 3": chandler
                }},
    "test_215": {"dialogue_id": "s01_e23_c10", "turns": [0,6],
                "speaker_map": {
                    "Speaker 3": "Carol Willick", "Speaker 4": rachel, "Speaker 5": "Dr. Franzblau" 
                }},
    "test_280": {"dialogue_id": "s04_e24_c19", "turns": [9,16],
                "speaker_map": {
                    "Speaker 1": "Andrea Waltham", "Speaker 2": phoebe
                }},
    "test_302": {"dialogue_id": "s09_e20_c02", "turns": [18,44],
                "speaker_map": {
                    "Speaker 1": ross, "Speaker 2": chandler, "Speaker 3": monica,
                    "Speaker 4": "Charlie", "Speaker 5": "Gunther", "Speaker 6": "Professor Spafford"
                }},
    "test_334": {"dialogue_id": "s06_e20_c11", "turns": [0,20],
                "speaker_map": {
                    "Speaker 1": chandler, "Speaker 2": joey
                }},
    "test_335": {"dialogue_id": "s07_e21_c03", "turns": [14,16],
                "speaker_map": {
                    "Speaker 1": ross, "Speaker 2": monica, "Speaker 3": chandler,
                }},
    "test_336": {"dialogue_id": "s09_e03_c13", "turns": [0,21],
                "speaker_map": {
                    "Speaker 1": "Mike Hannigan", "Speaker 2": phoebe
                }},

}

unmappable = [
    "train_21",# maps to s02_e22_c05 and s02_e22_c06
    "train_22",# maps to s10_e15_c09 and s10_e15_c11
    "train_39",# maps to s04_e13_c08 and s04_e13_c13
    "train_63",# maps to s02_e17_c05 and s02_e17_c06
    "train_154",# maps to s02_e07_c04 and s02_e07_c05
    "train_299",# maps to s01_e14_c02 and s01_e14_c05
    "train_356",# maps to s03_e11_c06 and s03_e11_c07
    "train_359",# maps to s05_e18_c04 and s05_e18_c05
    "train_409",# maps to s01_e22_c02 and s01_e22_c04
    "train_498",# maps to s02_e16_c02 and s02_e16_c03
    "train_504",# maps to s08_e19_c02, and s08_e19_c03 and has loads of"flashbacks" in between utterances
    "train_508",# maps to s08_e04_c01 and s08_e04_c03
    "train_552",# maps to s04_e05_c02 and s04_e05_c06
    "train_564",# maps to s04_e14_c04 and s04_e14_c05
    "train_614",# maps to s02_e03_c04 and s02_e03_c06
    "train_636",# maps to s03_e06_c02 and s03_e06_c03
    "train_664",# maps to s05_e17_c08 and s05_e17_c09
    "train_684",# maps to s03_e15_c01 and s03_e15_c02
    "train_698",# maps to s01_e14_c05 and s01_e14_c10
    "train_701",# maps to s10_e05_c02 and s10_e05_c03
    "train_706",# maps to s09_e10_c09 and s09_e10_c10
    "train_721",# maps to s03_e15_c04 and s03_e15_c09
    "train_795",# maps to s03_e05_c10 and s03_e05_c12
    "train_796",# maps to s05_e14_c08 and s05_e14_c09
    "train_801",# maps to s05_e08_c09 and s05_e08_c10
    "train_803",# maps to s10_e03_c10 and s10_e03_c12
    "train_810",# maps to s05_e09_c07 and s05_e09_c09
    "train_898",# maps to s06_e14_c04 and s06_e14_c09
    "train_966",# unknown dialogue that doesn't exist in emoryNLP
    "train_972",# maps to s10_e05_c01 and s10_e05_c07
    "train_973",# unknown dialogue that doesn't exist in emoryNLP
    "train_1008",# maps to s01_e16_c01 and s01_e16_c02
    "train_1058",# maps to s05_e13_c01 and s05_e13_c02
    "train_1072",# maps to s01_e12_c06 and s01_e12_c07
    "dev_31",# maps to s03_e19_c08 and s03_e19_c13
    "dev_101",# maps to s01_e15_c02 and s01_e15_c03
    "dev_132",# maps to s06_e04_c16 and s06_e04_c17
    "dev_136",# maps to s06_e03_c05 and s06_e03_c06
    "dev_144",# maps to s04_e20_c02 and s04_e20_c04
    "dev_202",# maps to s03_e02_c02 and s03_e02_c03
    "dev_222",# maps to s05_e13_c06 and s05_e13_c07
    "dev_245",# maps to s05_e19_c02 and s05_e19_c03
    "dev_276",# maps to s03_e13_c07 and s03_e13_c10
    "dev_291",# maps to s03_e06_c04 and s03_e06_c05
    "test_22",# maps to s05_e08_c06 and s05_e08_c09
    "test_44",# maps to s03_e20_c08 and s03_e20_c09
    "test_66",# unknown dialogue that doesn't exist in emoryNLP
    "test_86",# maps to s04_e04_c13 and s04_e04_c14
    "test_153",# maps to s09_e11_c02 and s09_e11_c04
    "test_165",# maps to s01_e13_c03 and s01_e13_c05
    "test_167",# maps to s03_e20_c05 and s03_e20_c06
    "test_197",# maps to s01_e14_c02 and s01_e14_c03
    "test_249",# maps to s02_e01_c03 and s02_e01_c04
    "test_305",# maps to s03_e16_c11 and s03_e16_c12
    "test_316",# maps to s06_e04_c03 and s06_e04_c05
]