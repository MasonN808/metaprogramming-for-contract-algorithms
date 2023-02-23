import pickle
from matplotlib import pyplot as plt

plt.style.use('seaborn-v0_8')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.rcParams['text.usetex'] = True

PLOT_TYPE = 'bar-graph'
SIMULATIONS = 150
NODES = [
    {'id': 0, 'label': r'$v_0$'},
    {'id': 1, 'label': r'$v_1$'},
    {'id': 2, 'label': r'$v_2$'},
    {'id': 3, 'label': r'$v_3$'},
    {'id': 4, 'label': r'$v_4$'},
    {'id': 5, 'label': r'$v_5$'},
    {'id': 6, 'label': r'$v_6$'},
    {'id': 7, 'label': r'$v_7$'},
    {'id': 8, 'label': r'$v_8$'},
    {'id': 9, 'label': r'$v_9$'},
    {'id': 10, 'label': r'$v_{10}$'},
    {'id': 11, 'label': r'$v_{11}$'},
    {'id': 12, 'label': r'$v_{12}$'},
    {'id': 13, 'label': r'$v_{13}$'},
    {'id': 14, 'label': r'$v_{14}$'}
]
TECHNIQUE_INDEX = 12

TIME_ALLOCATION_FILE = 'src/tests/robotics_domain/data/time_data.txt'
TIME_ALLOCATIONS = pickle.load(open('src/tests/robotics_domain/data/time_data.txt', 'rb'))
TIME_ALLOCATIONS.insert(7, {TECHNIQUE_INDEX: [0 for _ in range(SIMULATIONS)]})
TIME_ALLOCATIONS.insert(13, {TECHNIQUE_INDEX: [0 for _ in range(SIMULATIONS)]})


def plot():    
    average_time_allocations = [0 for _ in range(len(NODES))]

    for index in range(len(NODES)):
        for simulation in range(SIMULATIONS):
            average_time_allocations[index] += TIME_ALLOCATIONS[NODES[index]['id']][TECHNIQUE_INDEX][simulation]
        average_time_allocations[index] /= SIMULATIONS

    plt.figure(figsize=(12, 6))

    axis = plt.gca()
    plt.setp(axis.get_xticklabels(), fontsize=14)
    plt.setp(axis.get_yticklabels(), fontsize=14)

    plt.ylabel("Average Time Allocation [sec]")
    plt.xlabel("Vertex")

    plt.bar(x=[node['label'] for node in NODES], height=average_time_allocations)

    plt.tight_layout()
    plt.show()


def main():
    plot()


if __name__ == "__main__":
    main()