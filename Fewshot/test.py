def sample(n, k):
    q, r = divmod(n, k)
    counts = [q+1]*r + [q]*(k-r)
    return counts

n = 16  # total items
k = 5   # total categories

print(sample(n, k))  # Outputs: [4, 3, 3, 3, 3]