---
layout: leetcode-entry
title: "2257. Count Unguarded Cells in the Grid"
permalink: "/leetcode/problem/2024-11-21-2257-count-unguarded-cells-in-the-grid/"
leetcode_ui: true
entry_slug: "2024-11-21-2257-count-unguarded-cells-in-the-grid"
---

[2257. Count Unguarded Cells in the Grid](https://leetcode.com/problems/count-unguarded-cells-in-the-grid/description/) medium
[blog post](https://leetcode.com/problems/count-unguarded-cells-in-the-grid/solutions/6068264/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21112024-2257-count-unguarded-cells?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/bR2zLEM0Giw)
[deep-dive](https://notebooklm.google.com/notebook/6861d24c-642a-4369-a07b-562ff166fa9e/audio)
![1.webp](/assets/leetcode_daily_images/b09e8b96.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/807

#### Problem TLDR

Count unseen cells in 2D matrix with guards and walls #medium #matrix

#### Intuition

Two ways to cast a ray:
1. Cast left-right, up-down for each row/column
2. Cast in 4 direaction from each guard (sligthly faster)

#### Approach

* write explicit loops or iterate over directions
* use 2D or 1D support grid

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(nm)$$

#### Code

```kotlin

    fun countUnguarded(m: Int, n: Int, guards: Array<IntArray>, walls: Array<IntArray>): Int {
        val g = Array(m) { IntArray(n) }; var i = 0
        for ((y, x) in walls) g[y][x] = 2; for ((y, x) in guards) g[y][x] = 3
        for ((y, x) in guards) {
            i = y + 1; while (i < m && g[i][x] < 2) g[i++][x] = 1
            i = y - 1; while (i >= 0 && g[i][x] < 2) g[i--][x] = 1
            i = x + 1; while (i < n && g[y][i] < 2) g[y][i++] = 1
            i = x - 1; while (i >= 0 && g[y][i] < 2) g[y][i--] = 1
        }
        return g.sumOf { it.count { it < 1 } }
    }

```
```rust

    pub fn count_unguarded(m: i32, n: i32, guards: Vec<Vec<i32>>, walls: Vec<Vec<i32>>) -> i32 {
        let (m, n, mut i) = (m as usize, n as usize, 0); let mut g = vec![vec![0; n]; m];
        for c in walls { g[c[0] as usize][c[1] as usize] = 2 }
        for c in &guards { g[c[0] as usize][c[1] as usize] = 3 }
        for c in &guards { let (y, x) = (c[0] as usize, c[1] as usize);
            i = y + 1; while i < m && g[i][x] < 2 { g[i][x] = 1; i += 1 }
            i = y; while i > 0 && g[i - 1][x] < 2 { g[i - 1][x] = 1; i -= 1 }
            i = x + 1; while i < n && g[y][i] < 2 { g[y][i] = 1; i += 1 }
            i = x; while i > 0 && g[y][i - 1] < 2 { g[y][i - 1] = 1; i -= 1 }
        }
        g.iter().map(|r| r.iter().filter(|&&c| c < 1).count() as i32).sum()
    }

```
```c++

    int countUnguarded(int m, int n, vector<vector<int>>& guards, vector<vector<int>>& walls) {
        vector<int> g(m * n);
        for (auto& pos : walls) g[pos[0] * n + pos[1]] = 2;
        for (auto& pos : guards) g[pos[0] * n + pos[1]] = 2;
        for (auto& pos : guards)
            for (int i = 0, d[] = {1,0,-1,0,0,1,0,-1}; i < 7; i += 2)
                for (int y = pos[0] + d[i], x = pos[1] + d[i + 1];
                    y >= 0 && y < m && x >= 0 && x < n && g[y * n + x] < 2;)
                        g[y * n + x] = 1, y += d[i], x += d[i + 1];
        return count_if(g.begin(), g.end(), [](int v){ return v < 1; });
    }

```

