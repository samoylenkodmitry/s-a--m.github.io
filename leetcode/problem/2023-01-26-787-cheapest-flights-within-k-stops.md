---
layout: leetcode-entry
title: "787. Cheapest Flights Within K Stops"
permalink: "/leetcode/problem/2023-01-26-787-cheapest-flights-within-k-stops/"
leetcode_ui: true
entry_slug: "2023-01-26-787-cheapest-flights-within-k-stops"
---

[787. Cheapest Flights Within K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/description/) medium

[https://t.me/leetcode_daily_unstoppable/98](https://t.me/leetcode_daily_unstoppable/98)

[blog post](https://leetcode.com/problems/cheapest-flights-within-k-stops/solutions/3102372/kotlin-bellman-ford/)

```kotlin
    fun findCheapestPrice(n: Int, flights: Array<IntArray>, src: Int, dst: Int, k: Int): Int {
        var dist = IntArray(n) { Int.MAX_VALUE }
        dist[src] = 0
        repeat(k + 1) {
            val nextDist = dist.clone()
            flights.forEach { (from, to, price) ->
                if (dist[from] != Int.MAX_VALUE && dist[from] + price < nextDist[to])
                    nextDist[to] = dist[from] + price
            }
            dist = nextDist
        }
        return if (dist[dst] == Int.MAX_VALUE) -1 else dist[dst]
    }

```

#### Intuition
DFS and Dijkstra gives TLE.
As we need to find not just shortest path price, but only for `k` steps, naive Bellman-Ford didn't work.
Let's define `dist`, where `dist[i]` - the shortest distance from `src` node to `i`-th node.
We initialize it with `MAX_VALUE`, and `dist[src]` is 0 by definition.
Next, we walk exactly `k` steps, on each of them, trying to minimize price.
If we have known distance to node `a`, `dist[a] != MAX`.
And if there is a link to node `b` with `price(a,b)`, then we can optimize like this `dist[b] = min(dist[b], dist[a] + price(a,b))`.
Because we're starting from a single node `dist[0]`, we will increase distance only once per iteration.
So, making `k` iterations made our path exactly `k` steps long.

#### Approach
* by the problem definition, path length is `k+1`, not just `k`
* we can't optimize a path twice in a single iteration, because then it will overreach to the next step before the current is finished.
* That's why we only compare distance from the previous step.

Space: O(kE), Time: O(k)

