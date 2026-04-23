---
layout: leetcode-entry
title: "3342. Find Minimum Time to Reach Last Room II"
permalink: "/leetcode/problem/2025-05-08-3342-find-minimum-time-to-reach-last-room-ii/"
leetcode_ui: true
entry_slug: "2025-05-08-3342-find-minimum-time-to-reach-last-room-ii"
---

[3342. Find Minimum Time to Reach Last Room II](https://leetcode.com/problems/find-minimum-time-to-reach-last-room-ii/description/) medium
[blog post](https://leetcode.com/problems/find-minimum-time-to-reach-last-room-ii/solutions/6724762/kotlin-rust-by-samoylenkodmitry-af6d/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/08052025-3342-find-minimum-time-to?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/5WmAP7tNiJ0)
![1.webp](/assets/leetcode_daily_images/63a15c10.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/982

#### Problem TLDR

Fastest travel in graph, alternating dt=1,2 #medium #bfs

#### Intuition

The naive Dijkstra without heap would give TLE.
With heap it is accepted.

#### Approach

Some optimizations implementation details
* return result as fast as possible
* no need to store the best time and compare it if heap is already picks the best
* mark visited by mutating the graph (not production code, time golf)

#### Complexity

- Time complexity:
$$O(V + ElogV)$$

- Space complexity:
$$O(V)$$

#### Code

```kotlin

// 47ms https://leetcode.com/problems/find-minimum-time-to-reach-last-room-ii/submissions/1628359473
    fun minTimeToReach(g: Array<IntArray>): Int {
        val w = g[0].size - 1; val h = g.size - 1
        val q = PriorityQueue<IntArray>(compareBy { it[0] })
        val d = intArrayOf(0, 1, 0, -1, 0); q += intArrayOf(0, 0, 0, 1)
        while (q.size > 0) {
            val (t, x, y, dt) = q.poll()
            for (i in 0..3) {
                val y = y + d[i]; val x = x + d[i + 1]
                if (x !in 0..w || y !in 0..h || g[y][x] < 0) continue
                val t = dt + max(t, g[y][x]); if (x == w && y == h) return t
                g[y][x] = -1; q += intArrayOf(t, x, y, 3 - dt)
            }
        }
        return 0
    }

```
```rust

// 11ms https://leetcode.com/problems/find-minimum-time-to-reach-last-room-ii/submissions/1628358131
    pub fn min_time_to_reach(mut g: Vec<Vec<i32>>) -> i32 {
        let (mut w, mut h) = (g[0].len() - 1, g.len() - 1);
        let (mut d, mut q) = ([0, 1, 0, -1, 0], BinaryHeap::from([(0, 0, 0, 1)]));
        while let Some((t, x, y, dt)) = q.pop() {
            for i in 0..4 {
                let y = (y as i32 + d[i]) as usize; let x = (x as i32 + d[i + 1]) as usize;
                if x > w || y > h || g[y][x] < 0 { continue }
                let t = dt + (-t).max(g[y][x]); if x == w && y == h { return t }
                g[y][x] = -1; q.push((-t, x, y, 3 - dt))
            }
        } 0
    }

```
```c++

// 0ms https://leetcode.com/problems/find-minimum-time-to-reach-last-room-ii/submissions/1628349530
    int minTimeToReach(vector<vector<int>>& g) {
        priority_queue<tuple<int, int, int, int>, vector<tuple<int, int, int, int>>, greater<>> q;
        q.emplace(0, 0, 0, 1); int h = size(g) - 1, w = size(g[0]) - 1, d[] = {0, 1, 0, -1, 0};
        while (size(q)) {
            auto [t1, x1, y1, dt] = q.top(); q.pop();
            for (int i = 0; i < 4; ++i) {
                int x = x1 + d[i]; int y = y1 + d[i + 1];
                if (min(x, y) < 0 || x > w || y > h || g[y][x] < 0) continue;
                int t = dt + max(t1, g[y][x]); if (x == w && y == h) return t;
                g[y][x] = -1; q.emplace(t, x, y, 3 - dt);
            }
        } return 0;
    }

```

