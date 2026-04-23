---
layout: leetcode-entry
title: "1334. Find the City With the Smallest Number of Neighbors at a Threshold Distance"
permalink: "/leetcode/problem/2024-07-26-1334-find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/"
leetcode_ui: true
entry_slug: "2024-07-26-1334-find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance"
---

[1334. Find the City With the Smallest Number of Neighbors at a Threshold Distance](https://leetcode.com/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/description/) medium
[blog post](https://leetcode.com/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/solutions/5537457/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/26072024-1334-find-the-city-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/mihonSEPLbg)
![2024-07-26_08-59_1.webp](/assets/leetcode_daily_images/1e46d79f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/682

#### Problem TLDR

Node with minimum neighbors by `distanceThreshold` #medium #bfs #FloydWarshall

#### Intuition

There are only 100 nodes maximum, so we can try to find all neighbors for each node independently. Depth-First Search will not work: some nodes can be revisited with better shorter paths. So, let's use the Breadth-First Search.
![2024-07-26_08-08.webp](/assets/leetcode_daily_images/a0b42402.webp)

Another way is to use Floyd-Warshall algorithm.
https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm
Repeat exactly `n` times the optimization procedure of choosing the minimum for every `i`, `j`, `k`: `path[j][k] = min(path[j][k], path[j][i] + path[i][k])`.

#### Approach

Let's implement BFS in Kotlin, Floyd-Warshall in Rust.

#### Complexity

- Time complexity:
$$O(V^3E)$$ for `V` times BFS `EV^2`, $$O(E + V^3)$$ for Floyd-Warshall

- Space complexity:
$$O(V + E)$$ for BFS, $$O(V^2)$$ for Floyd-Warshall

#### Code

```kotlin

    fun findTheCity(n: Int, edges: Array<IntArray>, distanceThreshold: Int): Int {
        val g = mutableMapOf<Int, MutableList<Pair<Int, Int>>>()
        for ((a, b, w) in edges) {
            g.getOrPut(a) { mutableListOf() } += b to w
            g.getOrPut(b) { mutableListOf() } += a to w
        }
        val queue = ArrayDeque<Int>()
        return (n - 1 downTo 0).minBy { x ->
            val dist = IntArray(n) { distanceThreshold + 1 }
            dist[x] = 0; queue.add(x); var count = 1
            while (queue.size > 0) queue.removeFirst().let { curr ->
                g[curr]?.forEach { (next, w) ->
                    if (w + dist[curr] < dist[next]) {
                        if (dist[next] > distanceThreshold) count++
                        dist[next] = w + dist[curr]; queue.add(next)
                    }
                }
            }
            count
        }
    }

```
```rust

    pub fn find_the_city(n: i32, edges: Vec<Vec<i32>>, distance_threshold: i32) -> i32 {
        let n = n as usize; let mut dist = vec![vec![i32::MAX / 2; n]; n];
        for u in 0..n { dist[u][u] = 0 }
        for e in edges {
            dist[e[0] as usize][e[1] as usize] = e[2];
            dist[e[1] as usize][e[0] as usize] = e[2]
        }
        for i in 0..n { for j in 0..n { for k in 0..n {
            dist[j][k] = dist[j][k].min(dist[j][i] + dist[i][k])
        }}}
        (0..n).rev().min_by_key(|&u|
            (0..n).filter(|&v| dist[u][v] <= distance_threshold).count()).unwrap() as i32
    }

```

