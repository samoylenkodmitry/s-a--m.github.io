---
layout: leetcode-entry
title: "2290. Minimum Obstacle Removal to Reach Corner"
permalink: "/leetcode/problem/2024-11-28-2290-minimum-obstacle-removal-to-reach-corner/"
leetcode_ui: true
entry_slug: "2024-11-28-2290-minimum-obstacle-removal-to-reach-corner"
---

[2290. Minimum Obstacle Removal to Reach Corner](https://leetcode.com/problems/minimum-obstacle-removal-to-reach-corner/description/) hard
[blog post](https://leetcode.com/problems/minimum-obstacle-removal-to-reach-corner/solutions/6090897/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/28112024-2290-minimum-obstacle-removal?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/wDnaQD5j6YM)
[deep-dive](https://notebooklm.google.com/notebook/86cfab05-137b-4191-9d3c-a4b80e82f49f/audio)
![1.webp](/assets/leetcode_daily_images/6351bfc7.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/814

#### Problem TLDR

Min removals to travel first-last in 2D grid #hard #bfs #dijkstra

#### Intuition

We are interested in the shortest path through obstacles, so the go-to algorithm is the BFS, then we optimize it with Dijkstra by moving only improved paths.

This simple optimization is not enough, however. So, we have another one - use a PriorityQueue to peek the smallest obstacles paths first.

And another cool trick: the are only two types of paths to sort - completely free and ones with obstacles. Free paths must go first. We completely drop the PriorityQueue and just add to the front or to the back. (this is a 0-1 BFS https://codeforces.com/blog/entry/22276)

#### Approach

* some other small optimizations are possible: we can stop searching at the first arrival to the end
* we can use a two Queues instead of one

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(nm)$$

#### Code

```kotlin

    fun minimumObstacles(grid: Array<IntArray>): Int {
        val obs = Array(grid.size) { IntArray(grid[0].size) { Int.MAX_VALUE }}
        val q = ArrayDeque<List<Int>>(listOf(listOf(0, 0, 0)))
        while (q.size > 0) {
            val (y, x, o) = q.removeFirst()
            if (y !in 0..<grid.size || x !in 0..<grid[0].size) continue
            val n = grid[y][x] + o
            if (n < obs[y][x]) {
                obs[y][x] = n
                for (s in listOf(y - 1, x, n, y, x + 1, n, y + 1, x, n, y, x - 1, n)
                    .chunked(3)) if (grid[y][x] > 0) q += s else q.addFirst(s)
            }
        }
        return obs[grid.size - 1][grid[0].size - 1]
    }

```
```rust

    pub fn minimum_obstacles(grid: Vec<Vec<i32>>) -> i32 {
        let mut obs = vec![vec![i32::MAX; grid[0].len()]; grid.len()];
        let mut q = VecDeque::from_iter([(1, 1, 0)]);
        while let Some((y, x, o)) = q.pop_front() {
            if y < 1 || y > grid.len() || x < 1 || x > grid[0].len() { continue }
            let n = grid[y - 1][x - 1] + o;
            if n < obs[y - 1][x - 1] {
                obs[y - 1][x - 1] =  n;
                for s in [(y - 1, x, n), (y + 1, x, n), (y, x - 1, n), (y, x + 1, n)] {
                    if grid[y - 1][x - 1] > 0 { q.push_back(s); } else { q.push_front(s); }
                }
            }
        }; obs[grid.len() - 1][grid[0].len() - 1]
    }

```
```

    int minimumObstacles(vector<vector<int>>& g) {
        int m = g.size(), n = g[0].size();
        vector<vector<int>> obs(m, vector<int>(n, INT_MAX));
        deque<tuple<int, int, int>> q; q.emplace_back(0, 0, 0);
        vector<pair<int, int>>dxy..-1, 0}, {0, 1}, {1, 0}, {0, -1..; // replace . to {
        while (q.size()) {
            auto [y, x, o] = q.front(); q.pop_front();
            for (auto [dy, dx]: dxy) {
                int ny = y + dy, nx = x + dx;
                if (ny < 0 || ny >= m || nx < 0 || nx >= n || g[ny][nx] + o >= obs[ny][nx]) continue;
                int n = g[ny][nx] + o; obs[ny][nx] = n;
                if (g[ny][nx] > 0) q.emplace_back(ny, nx, n); else q.emplace_front(ny, nx, n);
            }
        } return obs[m - 1][n - 1];
    }

```

