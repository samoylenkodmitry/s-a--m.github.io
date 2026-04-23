---
layout: leetcode-entry
title: "3651. Minimum Cost Path with Teleportations"
permalink: "/leetcode/problem/2026-01-28-3651-minimum-cost-path-with-teleportations/"
leetcode_ui: true
entry_slug: "2026-01-28-3651-minimum-cost-path-with-teleportations"
---

[3651. Minimum Cost Path with Teleportations](https://leetcode.com/problems/minimum-cost-path-with-teleportations/description/) hard
[blog post](https://leetcode.com/problems/minimum-cost-path-with-teleportations/solutions/7531530/kotiln-rust-by-samoylenkodmitry-icup/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/28012026-3651-minimum-cost-path-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/JgrJMR2Neyk)

![85e6716f-65a8-4af7-a0a6-c82521ea5836 (1).webp](/assets/leetcode_daily_images/584a2bac.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1251

#### Problem TLDR

Min cost to walk 2D grid with free teleports #hard #dijkstra #dp

#### Intuition

Didn't solved.

```j
    // BFS?
    // A*
    // Dijkstra
    //
    // store individual k for each path
    //
    // moves only right-down,
    // can teleport backwards for some optimal paths
    //
    // we have to try all possible paths
    //
    // visited set is x,y,k
    //
    // to teleport we have to try all "lower" values
    //
    // 80*80=1600 or 10^3 can be O(n) scan for lowers?
    //
    // DP=DFS+cache? - no, we can visit twice with better value, so should be Dijkstra
    //
    // TLE
    //
    // TLE TLE TLE
    // TLE
```

To make Dijkstra work skip already done teleportations for each lower[k], where lower is indices of all sorted values.

The dp solution: do k layered relaxations of teleportations and walks right-bottom.

#### Approach

* why in Dijkstra we can skip teleportations individually for each k?
* why in Dp solution we updating by batches of equal values the minimum value so far?
* why in Dp solution we sort descending and going from bigger to lower value and updating min(dp)?

#### Complexity

- Time complexity:
$$O(n^2klog(n))$$ for Dijkstra and n^2k for DP

- Space complexity:
$$O(n^2k)$$ for Dijkstra, n^2 for dp

#### Code

```kotlin
// 1505ms
    fun minCost(g: Array<IntArray>, k: Int): Int {
        val q = PriorityQueue<IntArray> { a,b -> a[3]-b[3] }
        val d = Array(g.size) { Array(g[0].size) { IntArray(k+1) {1 shl 30}}}
        val sorted = (0..<g.size*g[0].size).sortedBy {g[it/g[0].size][it%g[0].size]}
        q += intArrayOf(0, 0, k, 0); val lower = IntArray(k+1)
        while (q.size > 0) {
            val (x,y,k,c) = q.poll()
            if (x==g[0].size-1&&y==g.size-1) return c
            for ((x,y) in arrayOf(x+1 to y, x to y+1))
                if (x < g[0].size && y < g.size&&d[y][x][k]>c+g[y][x]) {
                    q += intArrayOf(x,y,k,c+g[y][x]); d[y][x][k] = c+g[y][x]
                }
            if (k > 0) while (lower[k] < sorted.size) {
                val i = sorted[lower[k]]; val (ny,nx) = i/g[0].size to i%g[0].size
                if (g[ny][nx] > g[y][x]) break
                if (c < d[ny][nx][k-1]) { q += intArrayOf(nx,ny,k-1,c); d[ny][nx][k-1] = c }
                lower[k]++
            }
        }; return -1
    }
```
```rust
// 68ms
    pub fn min_cost(g: Vec<Vec<i32>>, k: i32) -> i32 {
        let w = g[0].len(); let g = g.concat(); let n = g.len();
        let mut d = vec![1 << 30; n]; d[0] = 0;
        let mut v: Vec<_> = (0..n).collect(); v.sort_by_key(|&i| -g[i]);
        for a in 0..=k {
            for i in 0..n {
                if i >= w { d[i] = d[i].min(d[i - w] + g[i]); }
                if i % w > 0 { d[i] = d[i].min(d[i - 1] + g[i]); }
            }
            if a == k { break; }
            let (mut m, mut s) = (1 << 30, 0);
            for i in 0..=n {
                if i == n || g[v[i]] != g[v[s]]
                    { for &j in &v[s..i] { d[j] = m; } s = i }
                if i < n { m = m.min(d[v[i]]); }
            }
        } d[n-1]
    }
```

