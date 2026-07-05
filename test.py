# Data
weights = [20, 30, 35, 80, 70]
distance = [2, 3, 5, 2, 6]

# Min-Max Scaling for weights
min_w = min(weights)
max_w = max(weights)

scaled_weights = []
for w in weights:
    scaled_weights.append((w - min_w) / (max_w - min_w))


# Min-Max Scaling for distance
min_d = min(distance)
max_d = max(distance)

scaled_distance = []
for d in distance:
    scaled_distance.append((d - min_d) / (max_d - min_d))

print("Scaled Weights:", scaled_weights)
print("Scaled Distance:", scaled_distance)