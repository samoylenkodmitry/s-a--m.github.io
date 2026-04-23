---
layout: leetcode-entry
title: "3341. Find Minimum Time to Reach Last Room I"
permalink: "/leetcode/problem/2025-05-07-3341-find-minimum-time-to-reach-last-room-i/"
leetcode_ui: true
entry_slug: "2025-05-07-3341-find-minimum-time-to-reach-last-room-i"
---

[3341. Find Minimum Time to Reach Last Room I](https://leetcode.com/problems/find-minimum-time-to-reach-last-room-i/description) medium
[blog post](https://leetcode.com/problems/find-minimum-time-to-reach-last-room-i/solutions/6722422/kotlin-rust-by-samoylenkodmitry-fx4w/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/07052025-3341-find-minimum-time-to?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/e_F70F5-iFQ)
![1.webp](/assets/leetcode_daily_images/c62f41a3.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/981

#### Problem TLDR

Fastest path in graph #medium #bfs

#### Intuition

Use BFS, track time.

#### Approach

* use time-improvement condition, or a Heap (heap is faster)

#### Complexity

- Time complexity:
$$O(V^2)$$, or (E + V)logV for heap

- Space complexity:
$$O(V)$$

#### Code

```kotlin

// 184ms
    fun minTimeToReach(moveTime: Array<IntArray>): Int {
        val w = moveTime[0].size; val time = IntArray(moveTime.size * w) { Int.MAX_VALUE }
        val q = ArrayDeque<Int>(); q += 0; time[0] = 0; val dxy = intArrayOf(-1, 1, -w, w)
        while (q.size > 0) {
            val xy = q.removeFirst(); val x1 = xy % w; val y1 = xy / w; val t = time[xy]
            for (d in dxy) {
                val xy2 = xy + d; val y2 = xy2 / w; val x2 = xy2 % w
                if (xy2 in 0..<time.size && (y1 == y2 || x1 == x2)) {
                    val t2 = 1 + max(t, moveTime[y2][x2])
                    if (t2 < time[xy2]) { time[xy2] = t2; q += xy2 }
                }
            }
        }
        return time.last()
    }

```

```kotlin

// 163ms https://leetcode.com/problems/find-minimum-time-to-reach-last-room-i/submissions/1627622531
    fun minTimeToReach(moveTime: Array<IntArray>): Int {
        val w = moveTime[0].size; val n = w * moveTime.size - 1
        val q = PriorityQueue<IntArray>(compareBy { it[1] })
        q += intArrayOf(0, 0); val dxy = intArrayOf(-1, 1, -w, w)
        while (q.size > 0) {
            val (xy, t) = q.poll(); val x1 = xy % w; val y1 = xy / w
            if (xy == n) return t
            for (d in dxy) {
                val xy2 = xy + d; val y2 = xy2 / w; val x2 = xy2 % w
                if (xy2 in 0..n && (y1 == y2 || x1 == x2) && moveTime[y2][x2] >= 0) {
                    q += intArrayOf(xy2, (1 + max(t, moveTime[y2][x2])))
                    moveTime[y2][x2] = -1
                }
            }
        }
        return -1
    }

```
```rust

// 0ms https://leetcode.com/problems/find-minimum-time-to-reach-last-room-i/submissions/1627635883
    pub fn min_time_to_reach(mut move_time: Vec<Vec<i32>>) -> i32 {
        let mut w = move_time[0].len(); let n = w * move_time.len() - 1;
        let mut q = BinaryHeap::from([(0, 0)]);
        let dxy = [-1, 1, -(w as i32), w as i32];
        while let Some((t, xy)) = q.pop() {
            let (x1, y1) = (xy % w, xy / w);
            if xy == n { return -t }
            for d in dxy {
                let xy2 = (xy as i32 + d) as usize; let (y2, x2) = (xy2 / w, xy2 % w);
                if xy2 <= n && (y1 == y2 || x1 == x2) && move_time[y2][x2] >= 0 {
                    q.push((-1 + t.min(-move_time[y2][x2]), xy2));
                    move_time[y2][x2] = -1
                }
            }
        } 0
    }

```
```c++

// 0ms https://leetcode.com/problems/find-minimum-time-to-reach-last-room-i/submissions/1627644831
    int minTimeToReach(vector<vector<int>>& g) {
        int h = size(g), w = size(g[0]), n = h * w - 1;
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> q;
        int dir[4] = {-1, 1, -w, w}; q.push({0, 0});
        while (size(q)) {
            auto [t, u] = q.top(); q.pop(); if (u == n) return t;
            for (int i = 0; i < 4; ++i) {
                int v = u + dir[i];
                if (v >= 0 && v <= n && (u / w == v / w || u % w == v % w) && g[v / w][v % w] >= 0)
                    q.push({max(t, g[v / w][v % w]) + 1, v}), g[v / w][v % w] = -1;
            }
        } return 0;
    }

```

