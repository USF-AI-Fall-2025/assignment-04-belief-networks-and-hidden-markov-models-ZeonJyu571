from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

car_model = DiscreteBayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition","Starts"),
        ("Gas","Starts"),
        ("Starts","Moves"),
])

# Defining the parameters using CPT


cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery":['Works',"Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas":['Full',"Empty"]},
)

cpd_radio = TabularCPD(
    variable=  "Radio", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable=  "Ignition", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[[0.95, 0.05, 0.05, 0.001], [0.05, 0.95, 0.95, 0.9999]],
    evidence=["Ignition", "Gas"],
    evidence_card=[2, 2],
    state_names={"Starts":['yes','no'], "Ignition":["Works", "Doesn't work"], "Gas":['Full',"Empty"]},
)

cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01],[0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no'] }
)


# Associating the parameters with the model structure
car_model.add_cpds( cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves)

car_infer = VariableElimination(car_model)

#print(car_infer.query(variables=["Moves"],evidence={"Radio":"turns on", "Starts":"yes"}))

def main():
    print("\n 1. Given that the car will not move, what is the probability that the battery is not working?")
    result1 = car_infer.query(["Battery"], {"Moves": "no"})
    P_not_B_given_not_M = result1.values[1]
    print(P_not_B_given_not_M)

    print("\n 2. Given that the radio is not working, what is the probability that the car will not start?")
    result2 = car_infer.query(["Starts"], {"Radio": "Doesn't turn on"})
    P_not_S_given_not_R = result2.values[1]
    print(P_not_S_given_not_R)

    print("\n 3. Given that the battery is working, does the probability of the radio working change if we discover that the car has gas in it?")
    print(" 1) P(R|B, G)")
    result3 = car_infer.query(["Radio"], {"Battery": "Works", "Gas": "Full"})
    P_R_given_B_and_G = result3.values[0]
    print(P_R_given_B_and_G)
    print(" 2) P(R|B, not G)")
    result4 = car_infer.query(["Radio"], {"Battery": "Works", "Gas": "Empty"})
    P_R_given_B_and_not_G = result4.values[0]
    print(P_R_given_B_and_not_G)
    print(" 3) Whether P(R|B, G) is equal to P(R|B)?")
    print(P_R_given_B_and_not_G == P_R_given_B_and_G)

    print("\n 4. Given that the car doesn't move, how does the probability of the ignition failing change if we observe that the car does not have gas in it?")
    print(" 1) P(not I|not M, not G)")
    result5 = car_infer.query(["Ignition"], {"Moves": "no", "Gas": "Empty"})
    P_not_I_given_not_M_and_not_G = result5.values[1]
    print(P_not_I_given_not_M_and_not_G)
    print(" 2) P(not I|not M, G)")
    result6 = car_infer.query(["Ignition"], {"Moves": "no", "Gas": "Full"})
    P_not_I_given_not_M_and_G = result6.values[1]
    print(P_not_I_given_not_M_and_G)
    print(" 3) Change")
    change = P_not_I_given_not_M_and_G - P_not_I_given_not_M_and_not_G
    if change == 0:
        print("The probability does not change")
    elif change > 0:
        print(f"The probability decreases {change}")
    else:
        print(f"The probability increases {abs(change)}")

    print("\n 5. What is the probability that the car starts if the radio works and it has gas in it?")
    result7 = car_infer.query(["Starts"], {"Radio": "turns on", "Gas": "Full"})
    P_S_given_R_and_G = result7.values[0]
    print(P_S_given_R_and_G)

if __name__ == "__main__":
    main()
