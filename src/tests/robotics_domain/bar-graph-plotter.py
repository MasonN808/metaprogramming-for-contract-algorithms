import pickle
from matplotlib import pyplot as plt

plt.style.use('seaborn-v0_8')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.rcParams['text.usetex'] = True

PLOT_TYPE = 'bar-graph'
SIMULATIONS = 150
TECHNIQUE_INDEX = 12
BUDGET =  10

NODES = {
    1: {'index': 1, 'label': r'$v_1$', 'color': 'seagreen'},
    2: {'index': 2, 'label': r'$v_2$', 'color': 'steelblue'},
    3: {'index': 3, 'label': r'$v_3$', 'color': 'seagreen'},
    4: {'index': 4, 'label': r'$v_4$', 'color': 'seagreen'},
    5: {'index': 5, 'label': r'$v_5$', 'color': 'seagreen'},
    6: {'index': 6, 'label': r'$v_6$', 'color': 'steelblue'},
    7: {'index': 8, 'label': r'$v_7$', 'color': 'indianred'},
    8: {'index': 9, 'label': r'$v_8$', 'color': 'indianred'},
    9: {'index': 10, 'label': r'$v_9$', 'color': 'orchid'},
    10: {'index': 11, 'label': r'$v_{10}$', 'color': 'orchid'},
    11: {'index': 12, 'label': r'$v_{11}$', 'color': 'orchid'},
    12: {'index': 14, 'label': r'$v_{12}$', 'color': 'indianred'},
}
NUM_NODES = 12
NODE_ORDERING = [12, 11, 10, 9, 8, 7, 5, 4, 3, 1, 6, 2]

TIME_ALLOCATION_FILE = 'src/tests/robotics_domain/data/time_data.txt'
TIME_ALLOCATIONS = pickle.load(open('src/tests/robotics_domain/data/time_data.txt', 'rb'))
TIME_ALLOCATIONS.insert(7, {TECHNIQUE_INDEX: [0 for _ in range(SIMULATIONS)]})
TIME_ALLOCATIONS.insert(13, {TECHNIQUE_INDEX: [0 for _ in range(SIMULATIONS)]})


def plot():    
    average_time_allocations = [0 for _ in range(NUM_NODES)]

    for id in range(NUM_NODES):
        for simulation in range(SIMULATIONS):
            average_time_allocations[id] += TIME_ALLOCATIONS[NODES[id + 1]['index']][TECHNIQUE_INDEX][simulation]
        average_time_allocations[id] /= SIMULATIONS

    total_time_allocations = sum(average_time_allocations)
    for i in range(len(average_time_allocations)):
        average_time_allocations[i] = BUDGET * (average_time_allocations[i] / total_time_allocations)

    plt.figure(figsize=(12, 6))

    axis = plt.gca()
    plt.setp(axis.get_xticklabels(), fontsize=14)
    plt.setp(axis.get_yticklabels(), fontsize=14)

    plt.ylabel("Average Time Allocation [sec]")
    plt.xlabel("Vertex")

    x = [NODES[id]['label'] for id in NODE_ORDERING]
    height = [average_time_allocations[id - 1] for id in NODE_ORDERING]
    color = [NODES[id]['color'] for id in NODE_ORDERING]
    plt.bar(x, height, color=color, hatch='//')

    for i in range(len(x)):
        plt.text(i, height[i] + 0.01, round(height[i], 2), ha='center', zorder=100)

    plt.tight_layout()
    plt.show()


def main():
    plot()


if __name__ == "__main__":
    main()