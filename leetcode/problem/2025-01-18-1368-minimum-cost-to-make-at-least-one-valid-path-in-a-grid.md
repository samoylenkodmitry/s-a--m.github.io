---
layout: leetcode-entry
title: "1368. Minimum Cost to Make at Least One Valid Path in a Grid"
permalink: "/leetcode/problem/2025-01-18-1368-minimum-cost-to-make-at-least-one-valid-path-in-a-grid/"
leetcode_ui: true
entry_slug: "2025-01-18-1368-minimum-cost-to-make-at-least-one-valid-path-in-a-grid"
---

[1368. Minimum Cost to Make at Least One Valid Path in a Grid](https://leetcode.com/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/description/) hard
[blog post](https://leetcode.com/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/solutions/6297925/kotlin-rust-by-samoylenkodmitry-tkv0/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18012025-1368-minimum-cost-to-make?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/g9MJ7RaEuGE)
![1.webp](/assets/leetcode_daily_images/b48ca51f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/868

#### Problem TLDR

Min undirected jumps to reach the end in grid #hard #bfs

#### Intuition

The naive Dijkstra with a sorted by `cost` queue will work out.

Some optimizations to make it `0-1 BFS`:
* use a simple non-sorted queue
* explore `free` movements first
* add `costly` movements to the end, keep track of the cost
* think of this like a minesweeper game, free islands got explored first, extra steps add cost

#### Approach

* we can use two queues: one for `free` and for `costly` movements, or just a single `Deque` adding to the front or back
* we can modify `grid` to track visited cells (golf solution, not a production code)

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(nm)$$

#### Code

```kotlin

    fun minCost(grid: Array<IntArray>): Int {
        val q = ArrayDeque<IntArray>(); q += intArrayOf(0, 0, 0)
        val d = listOf(0, 1, 0, -1, 1, 0, -1, 0)
        while (q.size > 0) {
            val (c, y, x) = q.removeFirst()
            if (y == grid.lastIndex && x == grid[0].lastIndex) return c
            if (grid.getOrNull(y)?.getOrNull(x) ?: 0 < 1) continue
            val curr = grid[y][x]; grid[y][x] = 0
            for (i in 0..3) if (i + 1 == curr)
                q.addFirst(intArrayOf(c, y + d[2 * i], x + d[2 * i + 1]))
                else q += intArrayOf(c + 1, y + d[2 * i], x + d[2 * i + 1])
        }
        return -1
    }

```
```rust

    pub fn min_cost(mut grid: Vec<Vec<i32>>) -> i32 {
        let mut q = VecDeque::from_iter([(0i32, 0i32, 0i32)]);
        let (m, n) = (grid.len() as i32, grid[0].len() as i32);
        while let Some((c, y, x)) = q.pop_front() {
            if y == m - 1 && x == n - 1 { return c }
            if !(0..m).contains(&y) || !(0..n).contains(&x) { continue }
            let curr = grid[y as usize][x as usize] as usize;
            grid[y as usize][x as usize] = 0;  if curr < 1 { continue }
            for (d, dy, dx) in [(1, 0, 1), (2, 0, -1), (3, 1, 0), (4, -1, 0)] {
                if d == curr { q.push_front((c, y + dy, x + dx)) }
                else { q.push_back((c + 1, y + dy, x + dx)) }}
        }; -1
    }

```
```c++

    int minCost(vector<vector<int>>& g) {
        deque<tuple<int, int, int>> q{ { {0, 0, 0} } };
        int d[] = {0, 1, 0, -1, 1, 0, -1, 0}, m = g.size(), n = g[0].size();
        while (q.size()) {
            auto [c, y, x] = q.front(); q.pop_front();
            if (y == m - 1 && x == n - 1) return c;
            if (min(y, x) < 0 || y == m || x == n || !g[y][x]) continue;
            int curr = exchange(g[y][x], 0);
            for (int i = 0; i < 4; ++i) if (i + 1 == curr)
                q.emplace_front(c, y + d[2 * i], x + d[2 * i + 1]); else
                q.emplace_back(c + 1, y + d[2 * i], x + d[2 * i + 1]);
        }
        return -1;
    }

```

