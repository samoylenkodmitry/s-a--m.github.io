---
layout: leetcode-entry
title: "2658. Maximum Number of Fish in a Grid"
permalink: "/leetcode/problem/2025-01-28-2658-maximum-number-of-fish-in-a-grid/"
leetcode_ui: true
entry_slug: "2025-01-28-2658-maximum-number-of-fish-in-a-grid"
---

[2658. Maximum Number of Fish in a Grid](https://leetcode.com/problems/maximum-number-of-fish-in-a-grid/description/) medium
[blog post](https://leetcode.com/problems/maximum-number-of-fish-in-a-grid/solutions/6338968/kotlin-rust-by-samoylenkodmitry-eptu/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/28012025-2658-maximum-number-of-fish?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/sHWNP68Sk5E)
![1.webp](/assets/leetcode_daily_images/b448ef19.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/878

#### Problem TLDR

Largest region in 2D grid #medium #bfs #dfs

#### Intuition

Do Depth/Breadth-First Search from any cell with fish.

#### Approach

* we can reuse grid to mark visited cells

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(nm)$$

#### Code

```kotlin

    fun findMaxFish(g: Array<IntArray>) =
        (0..<g.size * g[0].size).maxOf { yx ->
            fun dfs(y: Int, x: Int): Int =
                if (min(y, x) < 0 || y == g.size || x == g[0].size || g[y][x] < 1) 0
                else g[y][x].also { g[y][x] = 0 } +
                    dfs(y - 1, x) + dfs(y + 1, x) + dfs(y, x - 1) + dfs(y, x + 1)
            dfs(yx / g[0].size, yx % g[0].size)
        }

```
```rust

    pub fn find_max_fish(mut g: Vec<Vec<i32>>) -> i32 {
        (0..g.len() * g[0].len()).map(|i| { let mut s = 0;
            let mut q = VecDeque::from([(i / g[0].len(), i % g[0].len())]);
            while let Some((y, x)) = q.pop_front() { if g[y][x] > 0 {
                s += g[y][x]; g[y][x] = 0;
                if y > 0 { q.push_back((y - 1, x))}
                if x > 0 { q.push_back((y, x - 1))}
                if y < g.len() - 1 { q.push_back((y + 1, x))}
                if x < g[0].len() - 1 { q.push_back((y, x + 1))}
            }}; s
        }).max().unwrap()
    }

```
```c++

    int findMaxFish(vector<vector<int>>& g) {
        int n = size(g), m = size(g[0]), r = 0;
        auto d = [&](this auto const& d, int y, int x) -> int {
            return min(y, x) < 0 || y == n || x == m || !g[y][x] ? 0 :
            exchange(g[y][x], 0) + d(y - 1, x) + d(y + 1, x) + d(y, x - 1) + d(y, x + 1);
        };
        for (int i = 0; i < m * n; ++i) r = max(r, d(i / m, i % m));
        return r;
    }

```

