from itertools import product

DEEP_BLOCK_OUTPUT = [16,24,32]
NUM_HEADS = [[1,1,1],[2,2,2],[4,4,4]]
GAT_layers = [[4,16,16], [16,32,32]]

hp_sets = product(DEEP_BLOCK_OUTPUT, NUM_HEADS, GAT_layers)
hp_sets = [hpset for hpset in hp_sets]
print("Number of hp sets: ",  len(hp_sets))
for i, hpset in enumerate(hp_sets):
    print(f"hpset {i}: {hpset}")

# Hyperparameters to test using Cross-Validation: Deep Block Output; Number of Heads; Number of GAT Layers
# hpset 0: (16, [1, 1, 1], [4, 16, 16])
# hpset 1: (16, [1, 1, 1], [16, 32, 32])
# hpset 2: (16, [2, 2, 2], [4, 16, 16])
# hpset 3: (16, [2, 2, 2], [16, 32, 32])
# hpset 4: (16, [4, 4, 4], [4, 16, 16])
# hpset 5: (16, [4, 4, 4], [16, 32, 32])
# hpset 6: (24, [1, 1, 1], [4, 16, 16])
# hpset 7: (24, [1, 1, 1], [16, 32, 32])
# hpset 8: (24, [2, 2, 2], [4, 16, 16])
# hpset 9: (24, [2, 2, 2], [16, 32, 32])
# hpset 10: (24, [4, 4, 4], [4, 16, 16])
# hpset 11: (24, [4, 4, 4], [16, 32, 32])
# hpset 12: (32, [1, 1, 1], [4, 16, 16])
# hpset 13: (32, [1, 1, 1], [16, 32, 32])
# hpset 14: (32, [2, 2, 2], [4, 16, 16])
# hpset 15: (32, [2, 2, 2], [16, 32, 32])
# hpset 16: (32, [4, 4, 4], [4, 16, 16])
# hpset 17: (32, [4, 4, 4], [16, 32, 32])