def get_simulation_case():
    simulation_list = []
    # Ev1: standard
    td_sim = [9, 17, 32, 42, 57, 64, 79, 89, 105, 115, 131, 136, 154, 162, 167]  # depature time
    ta_sim = [1, 11, 19, 34, 43, 58, 66, 81, 91, 106, 116, 132, 137, 156, 163]  # start charging time
    tx_sim = [7, 14, 30, 38, 54, 62, 76, 88, 101, 112, 128, 133, 150, 160, 165]  # anxious time
    temp = [td_sim, ta_sim, tx_sim]
    simulation_list.append(temp)
    # Ev2: go to work early
    td_sim = [8, 15, 31, 38, 54, 63, 79, 88, 103, 111, 131, 134, 154, 160, 167]  # depature time
    ta_sim = [1, 10, 16, 33, 40, 56, 64, 81, 89, 104, 113, 132, 135, 155, 161]  # start charging time
    tx_sim = [6, 13, 29, 37, 42, 61, 77, 86, 101, 110, 129, 133, 152, 158, 165]  # anxious time
    temp = [td_sim, ta_sim, tx_sim]
    simulation_list.append(temp)
    # Ev3: inclined to work overtime
    td_sim = [11, 21, 34, 46, 59, 70, 83, 94, 108, 117, 132, 135, 155, 159, 167]  # depature time
    ta_sim = [1, 12, 22, 35, 47, 60, 71, 84, 95, 109, 118, 133, 136, 156, 160]  # start charging time
    tx_sim = [7, 18, 32, 43, 55, 67, 81, 91, 105, 114, 129, 133, 151, 157, 164]  # anxious time
    temp = [td_sim, ta_sim, tx_sim]
    simulation_list.append(temp)
    for i in range(1000):
        td_sim = [9, 17, 32, 42, 57, 64, 79, 89, 105, 115, 131, 136, 154, 162, 167]  # depature time
        ta_sim = [1, 11, 19, 34, 43, 58, 66, 81, 91, 106, 116, 132, 137, 156, 163]  # start charging time
        tx_sim = [7, 14, 30, 38, 54, 62, 76, 88, 101, 112, 128, 133, 150, 160, 165]  # anxious time
        temp = [td_sim, ta_sim, tx_sim]
        simulation_list.append(temp)

    return simulation_list


def get_EV_case():
    EV_case = []
    mu_arr = [18.8, 9.8, 11.6]
    sigma_arr = [1.6, 1.9, 1.6]
    mu_dep = [9.1, 18.4, 15.6]
    sigma_dep = [1.3, 1.1, 1.7]
    temp = [mu_arr, sigma_arr, mu_dep, sigma_dep]
    EV_case.append(temp)

    mu_arr = [16.8, 9.0, 11.6]
    sigma_arr = [1.6, 1.9, 1.6]
    mu_dep = [7.2, 15.4, 15.6]
    sigma_dep = [1.2, 1.1, 1.7]
    temp = [mu_arr, sigma_arr, mu_dep, sigma_dep]
    EV_case.append(temp)

    mu_arr = [22.5, 11.6, 11.6]
    sigma_arr = [1.6, 1.5, 1.6]
    mu_dep = [10.8, 21.4, 15.6]
    sigma_dep = [1.2, 1.8, 1.7]
    temp = [mu_arr, sigma_arr, mu_dep, sigma_dep]
    EV_case.append(temp)
    for i in range(1000):
        mu_arr = [18.8, 9.8, 11.6]
        sigma_arr = [1.6, 1.9, 1.6]
        mu_dep = [9.1, 18.4, 15.6]
        sigma_dep = [1.3, 1.1, 1.7]
        temp = [mu_arr, sigma_arr, mu_dep, sigma_dep]
        EV_case.append(temp)

    return EV_case
