---
layout: leetcode-entry
title: "3650. Minimum Cost Path with Edge Reversals"
permalink: "/leetcode/problem/2026-01-27-3650-minimum-cost-path-with-edge-reversals/"
leetcode_ui: true
entry_slug: "2026-01-27-3650-minimum-cost-path-with-edge-reversals"
---

[3650. Minimum Cost Path with Edge Reversals](https://leetcode.com/problems/minimum-cost-path-with-edge-reversals/description/) medium
[blog post](https://leetcode.com/problems/minimum-cost-path-with-edge-reversals/solutions/7528684/kotlin-rust-by-samoylenkodmitry-gp5i/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27012026-3650-minimum-cost-path-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/2xGssuWoFMc)

![91f5b752-fce8-4bd7-9d94-09b6bc56809d (1).webp](/assets/leetcode_daily_images/c834395c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1250

#### Problem TLDR

Min path in weighted graph with reverse 2w nodes #medium #dijkstra

#### Intuition

Add reversed nodes to the graph.
The *reverse only once* rule is automatically handled by Dijkstra: the optimal path will not visit nodes twice anyway.

```j
    // bfs? how to take reversal into account?
    //      how to choose which node to reverse?
    // dijkstra
    //
    // how to handle immediate traversal?
    // just replace original node with reversed dest?
    //
    // how to reverse just *once*?
    //
```

#### Approach

* we don't have to track distance array, the visited flags + heap is enough

#### Complexity

- Time complexity:
$$O((v+e)logV)$$

- Space complexity:
$$O(v^2)$$

#### Code

```kotlin
// 124ms
    fun minCost(n: Int, e: Array<IntArray>): Int {
        val g = Array(n) {ArrayList<Int>()}; val v = IntArray(n)
        for ((a,b,w) in e) { g[a] += w*n+b; g[b] += 2*w*n+a }
        val q = PriorityQueue<Long>(); q += 0
        return (0..4*n).firstNotNullOfOrNull {
            val wi = q.poll()?:0; val i = (wi%n).toInt()
            if (v[i]++==0) for (cj in g[i]) q += wi-i + cj
            if (i == n-1) (wi/n).toInt() else null
        } ?: -1
    }
```
```rust
// 64ms
    pub fn min_cost(n: i32, e: Vec<Vec<i32>>) -> i32 {
        let (N, mut g, mut q, mut v) = (n as i64, vec![vec![]; n as usize],
                                        BinaryHeap::from([0]), vec![0; n as usize]);
        for x in e {
            let (u, k, w) = (x[0] as usize, x[1] as usize, x[2] as i64);
            g[u].push(w * N + k as i64); g[k].push(w * 2 * N + u as i64);
        }
        while let Some(c) = q.pop() {
            let i = (-c % N) as usize;
            if i == v.len() - 1 { return (-c / N) as i32; }
            if v[i] == 0 { v[i] = 1; for x in &g[i] { q.push(c - c % N - x); }}
        } -1
    }
```

