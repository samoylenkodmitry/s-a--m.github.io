---
layout: leetcode-entry
title: "2577. Minimum Time to Visit a Cell In a Grid"
permalink: "/leetcode/problem/2024-11-29-2577-minimum-time-to-visit-a-cell-in-a-grid/"
leetcode_ui: true
entry_slug: "2024-11-29-2577-minimum-time-to-visit-a-cell-in-a-grid"
---

[2577. Minimum Time to Visit a Cell In a Grid](https://leetcode.com/problems/minimum-time-to-visit-a-cell-in-a-grid/description/) hard
[blog post](https://leetcode.com/problems/minimum-time-to-visit-a-cell-in-a-grid/solutions/6093910/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/29112024-2577-minimum-time-to-visit?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/KItOmpgESTA)
[deep-dive](https://notebooklm.google.com/notebook/84e7df30-11be-470e-9442-38dd6e7a1410/audio)
![1.webp](/assets/leetcode_daily_images/1471bd83.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/815

#### Problem TLDR

Min time start-end 4-d travel in 2D matrix with waiting #hard #dijkstra

#### Intuition

Start with simple BFS.
We can `wait` by moving back and forward incrementing time by `2`.
If we put k

#### Approach

* we can use a simple boolean visited set instead of comparing the time, as we always reach the earliest time first

#### Complexity

- Time complexity:
$$(nmlog(nm))$$

- Space complexity:
$$mn$$

#### Code

```kotlin

    fun minimumTime(grid: Array<IntArray>): Int {
        if (grid[0][1] > 1 && grid[1][0] > 1) return -1
        val q = PriorityQueue<List<Int>>(compareBy { it[2] }); q += listOf(0, 0, 0)
        val visited = Array(grid.size) { BooleanArray(grid[0].size)}
        while (q.size > 0) {
            val (y, x, t) = q.poll()
            if (y == grid.size - 1 && x == grid[0].size - 1) return t
            if (visited[y][x]) continue; visited[y][x] = true
            for ((y1, x1) in listOf(y - 1 to x, y to x + 1, y + 1 to x, y to x - 1))
                if (y1 in grid.indices && x1 in grid[0].indices && !visited[y1][x1])
                    q += listOf(y1, x1, 1 + max(grid[y1][x1] - max(0, grid[y1][x1] - t) % 2, t))
        }; return -1
    }

```

```rust

    pub fn minimum_time(grid: Vec<Vec<i32>>) -> i32 {
        if grid[0][1] > 1 && grid[1][0] > 1 { return -1 }
        let mut h = BinaryHeap::from_iter([(0, 1, 1)]);
        let mut time = vec![vec![i32::MAX; grid[0].len()]; grid.len()];
        while let Some((t, y, x)) = h.pop() {
            for (y1, x1) in [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)] {
                if y1.min(x1) < 1 || y1 > grid.len() || x1 > grid[0].len() { continue }
                let t = (-t + 1).max(grid[y1 - 1][x1 - 1] + (grid[y1 - 1][x1 - 1] + t + 1) % 2);
                if t < time[y1 - 1][x1 - 1] {  time[y1 - 1][x1 - 1] = t; h.push((-t, y1, x1)); }
            }
        }; time[grid.len() - 1][grid[0].len() - 1]
    }

```
```

    int minimumTime(vector<vector<int>>& g) {
        if (min(g[0][1], g[1][0]) > 1) return -1;
        priority_queue<array<int, 3>> pq; pq.push({0, 0, 0});
        vector<vector<int>> time(g.size(), vector<int>(g[0].size(), INT_MAX));
        while (pq.size()) {
            auto [t, y, x] = pq.top(); pq.pop();
            for (auto [y1, x1] : array<int[2],4>..{y - 1, x}, {y + 1, x}, {y, x - 1}, {y, x + 1\}..) { // replace '.' to '{'
                if (min(y1, x1) < 0 || y1 >= g.size() || x1 >= g[0].size()) continue;
                int t1 = max(-t + 1, g[y1][x1] + (g[y1][x1] + t + 1) % 2);
                if (t1 >= time[y1][x1]) continue;
                time[y1][x1] = t1; pq.push({-t1, y1, x1});
            }
        } return time.back().back();
    }

```

