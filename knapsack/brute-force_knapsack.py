""""
Simple brute-force implementation to solve the Knapsack problem
"""

#%%
import numpy as np


# Problem Definition
items_values = {"⚽️": 8, "💻": 47, "📸": 10, "📚": 5, "🎸": 16}
values_list = [8, 47, 10, 5, 16]
items_weight = {"⚽️": 3, "💻": 11, "📸": 14, "📚": 19, "🎸": 5}
weights_list = [3, 11, 14, 19, 5]
maximum_weight = 26


# For each items there is 2 options: 1 (we take the item) or 0 (we don't take the item)
# We then have 2^n combinations


def sum_weight(bitstring, items_weight):
    weight = 0
    for n, i in enumerate(items_weight):
        if bitstring[n] == "1":
            weight += i
    return weight


def sum_values(bitstring, items_value):
    value = 0
    for n, i in enumerate(items_value):
        if bitstring[n] == "1":
            value += i
    return value


items = list(items_values.keys())
n_items = len(items)
combinations = {}
max_value = 0

for case_i in range(2**n_items):  # all possible options
    
    combinations[case_i] = {}
    bitstring = np.binary_repr(case_i, n_items)  # bitstring representation of a possible combination, e.g, "01100" in our problem means bringing (-💻📸--)
    combinations[case_i]["items"] = [items[n] for n, i in enumerate(bitstring) if i == "1"]
    combinations[case_i]["value"] = sum_values(bitstring, values_list)
    combinations[case_i]["weight"] = sum_values(bitstring, weights_list)

    # save the information of the optimal solution (the one that maximizes the value while respecting the maximum weight)
    if (combinations[case_i]["value"] > max_value
        and combinations[case_i]["weight"] <= maximum_weight):

        max_value = combinations[case_i]["value"]
        optimal_solution = {"items": combinations[case_i]["items"],
                            "value": combinations[case_i]["value"],
                            "weight": combinations[case_i]["weight"]}


print(f"The best combination is {optimal_solution['items']} with a total value:\
        {optimal_solution['value']} and total weight {optimal_solution['weight']}")
# %%
