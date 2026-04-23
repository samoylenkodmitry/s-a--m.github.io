---
layout: leetcode-entry
title: "827. Making A Large Island"
permalink: "/leetcode/problem/2025-01-31-827-making-a-large-island/"
leetcode_ui: true
entry_slug: "2025-01-31-827-making-a-large-island"
---

[827. Making A Large Island](https://leetcode.com/problems/making-a-large-island/description/) hard
[blog post](https://leetcode.com/problems/making-a-large-island/solutions/6351552/kotlin-rust-by-samoylenkodmitry-cqr5/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/31012025-827-making-a-large-island?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/UGR9lhmy4wM)
![1.webp](/assets/leetcode_daily_images/d79eda0a.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/881

#### Problem TLDR

Max area after filling one empty 2D grid cell #hard #dfs #union_find

#### Intuition

Let's try all empty cells.
To quickly calculate the area, we have to precompute it using Union-Find or Depth-First Search with group counting.

#### Approach

* dfs code is shorter
* the edge case is when there are none empty cells
* use groups length as groups' counter, mark visited cells with it
* for Union-Find size check, careful to which parent the size goes
* filter the same group in different directions (use set or just check the list of four id values)
* don't rewrite input arguments memory in production code

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(nm)$$

#### Code

```kotlin

    fun largestIsland(g: Array<IntArray>): Int {
        val sz = mutableListOf(0, 0); var res = 0
        fun dfs(y: Int, x: Int): Int =
            if (y !in 0..<g.size || x !in 0..<g[0].size || g[y][x] != 1) 0 else {
            g[y][x] = sz.size; 1 + listOf(y - 1 to x, y + 1 to x, y to x - 1, y to x + 1)
            .sumOf { (y, x) -> dfs(y, x) }}
        for (y in g.indices) for (x in g[0].indices) if (g[y][x] < 1)
            res = max(res, 1 + listOf(y - 1 to x, y + 1 to x, y to x - 1, y to x + 1)
                .filter { (y, x) -> y in 0..<g.size && x in 0..<g[0].size}
                .map { (y, x) -> if (g[y][x] == 1) { sz += dfs(y, x) }; g[y][x] }
                .toSet().sumOf { sz[it] })
        return max(res, dfs(0, 0))
    }

```
```rust

    pub fn largest_island(mut g: Vec<Vec<i32>>) -> i32 {
        let (mut res, mut m, mut n) = (0, g.len(), g[0].len());
        let mut u: Vec<_> = (0..m * n).collect();
        let mut sz: Vec<_> = (0..m * n).map(|i| g[i / n][i % n] as usize).collect();
        let mut f = |a: usize, u: &mut Vec<usize>| { while u[a] != u[u[a]] { u[a] = u[u[a]]}; u[a] };
        let mut conn = |a: usize, b: usize, u: &mut Vec<usize>| {
            let (a, b) = (f(a, u), f(b, u));
            if a != b { u[a] = b; let s = sz[a]; sz[b] += s; sz[a] = 0 }};
        for y in 0..m { for x in 0..n { if g[y][x] > 0 {
            if y > 0 && g[y - 1][x] > 0 { conn((y - 1) * n + x, y * n + x, &mut u) }
            if x > 0 && g[y][x - 1] > 0 { conn(y * n + x - 1, y * n + x, &mut u) }
        }}}
        for y in 0..m { for x in 0..n { if g[y][x] < 1 {
            let mut fs: Vec<_> = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
                .into_iter().filter(|&(y, x)| y.min(x) >= 0 && y < m && x < n)
                .map(|(y, x)| f(y * n + x, &mut u)).collect(); fs.sort(); fs.dedup();
            res = res.max(1 + fs.iter().map(|&a| sz[a]).sum::<usize>())
        }}}
        res.max(sz[f(0, &mut u)]) as i32
    }

```
```c++

    int largestIsland(vector<vector<int>>& g) {
        int res = 0; vector<int> sz{0, 0};
        auto d = [&](this const auto& d, int y, int x) {
            if (min(y, x) < 0 || x >= size(g[0]) || y >= size(g) || g[y][x] != 1) return 0;
            g[y][x] = size(sz);
            return 1 + d(y - 1, x) + d(y + 1, x) + d(y, x - 1) + d(y, x + 1);
        };
        for (int y = 0; y < size(g); ++y) for (int x = 0; x < size(g[0]); ++x) if (!g[y][x]) {
            int sum = 1, s[4]{}, k = 0;
            for (auto [dy, dx]: {pair{-1, 0}, {1, 0}, {0, -1}, {0, 1}}) {
                int ny = y + dy, nx = x + dx;
                if (min(nx, ny) >= 0 && nx < size(g[0]) && ny < size(g)) {
                    if (g[ny][nx] == 1) sz.push_back(d(ny, nx));
                    if (find(s, s + k, g[ny][nx]) == s + k)
                        s[k++] = g[ny][nx], sum += sz[g[ny][nx]], res = max(res, sum);
                }
            }
        }
        return res ? res : size(g) * size(g[0]);
    }

```

