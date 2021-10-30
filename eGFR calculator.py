def eGFR (age, gender, cre):
    cre = cre*0.011312217194570135  ### convert umol/L to mg/dL ###
    if gender == 0:
        k = 0.7
        alpha = -0.329
        factor = 1.018
    if gender == 1:
        k = 0.9
        alpha = -0.411
        factor = 1
    eGFR = 141 * ((min(cre/k, 1)) ** alpha) * ((max (cre/k, 1)) **(-1.209)) * (0.993 ** age) * factor
    return print(eGFR)
