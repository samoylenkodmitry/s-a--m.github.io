---
layout: leetcode-entry
title: "2684. Maximum Number of Moves in a Grid"
permalink: "/leetcode/problem/2024-10-29-2684-maximum-number-of-moves-in-a-grid/"
leetcode_ui: true
entry_slug: "2024-10-29-2684-maximum-number-of-moves-in-a-grid"
---

[2684. Maximum Number of Moves in a Grid](https://leetcode.com/problems/maximum-number-of-moves-in-a-grid/description/) medium
[blog post](https://leetcode.com/problems/maximum-number-of-moves-in-a-grid/solutions/5981633/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/29102024-2684-maximum-number-of-moves?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/L8JW6qCp604)
[deep-dive](https://notebooklm.google.com/notebook/5df8bab2-e6b8-4d6e-afd5-42299f1fee20/audio)
![1.webp](/assets/leetcode_daily_images/47783397.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/784

#### Problem TLDR

Max increasing path from left to right in 2D matrix #medium #dynamic_programming

#### Intuition

On each cell we only care about three: `left-top`, `left` and `left-bottom`. Save the longest path so-far somewhere and increase if the condition met.

#### Approach

* corner case is when previous cell has zero path length, mitigate this with INT_MIN

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(nm)$$, can be optimized to just two columns O(n)

#### Code

```kotlin

    fun maxMoves(grid: Array<IntArray>): Int {
        val moves = Array(grid.size) { IntArray(grid[0].size)}
        var res = 0
        for (x in 1..<grid[0].size) for (y in grid.indices) {
            val v = grid[y][x]
            val a = if (y > 0 && v > grid[y - 1][x - 1])
                1 + moves[y - 1][x - 1] else Int.MIN_VALUE
            val b = if (v > grid[y][x - 1])
                1 + moves[y][x - 1] else Int.MIN_VALUE
            val c = if (y < grid.lastIndex && v > grid[y + 1][x - 1])
                1 + moves[y + 1][x - 1] else Int.MIN_VALUE
            moves[y][x] = maxOf(a, b, c); res = max(res, moves[y][x])
        }
        return res
    }

```
```rust

    pub fn max_moves(grid: Vec<Vec<i32>>) -> i32 {
        let (mut m, mut res) = (vec![vec![0; grid[0].len()]; grid.len()], 0);
        for x in 1..grid[0].len() { for y in 0..grid.len() {
            let v = grid[y][x];
            let a = if y > 0 && v > grid[y - 1][x - 1]
                { 1 + m[y - 1][x - 1] } else { i32::MIN };
            let b = if v > grid[y][x - 1]
                { 1 + m[y][x - 1] } else { i32::MIN };
            let c = if y < grid.len() - 1 && v > grid[y + 1][x - 1]
                { 1 + m[y + 1][x - 1] } else { i32::MIN };
            let r = a.max(b).max(c); m[y][x] = r; res = res.max(r)
        }}; res
    }

```
```c++

    int maxMoves(vector<vector<int>>& grid) {
        vector<vector<int>> m(grid.size(), vector<int>(grid[0].size(), 0));
        int res = 0;
        for (int x = 1; x < m[0].size(); ++x) for (int y = 0; y < m.size(); ++y) {
            int v = grid[y][x];
            int a = y > 0 && v > grid[y - 1][x - 1] ? 1 + m[y - 1][x - 1] : INT_MIN;
            int b = v > grid[y][x - 1] ? 1 + m[y][x - 1] : INT_MIN;
            int c = y < grid.size() - 1 && v > grid[y + 1][x - 1] ? 1 + m[y + 1][x - 1] : INT_MIN;
            m[y][x] = max(a, max(b, c)); res = max(res, m[y][x]);
        }
        return res;
    }

```

