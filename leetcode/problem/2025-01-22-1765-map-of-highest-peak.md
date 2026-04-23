---
layout: leetcode-entry
title: "1765. Map of Highest Peak"
permalink: "/leetcode/problem/2025-01-22-1765-map-of-highest-peak/"
leetcode_ui: true
entry_slug: "2025-01-22-1765-map-of-highest-peak"
---

[1765. Map of Highest Peak](https://leetcode.com/problems/map-of-highest-peak/description/) medium
[blog post](https://leetcode.com/problems/map-of-highest-peak/solutions/6314422/kotlin-rust-by-samoylenkodmitry-s0tr/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22012025-1765-map-of-highest-peak?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/XuWNOSmVgU0)
![1.webp](/assets/leetcode_daily_images/200a59f2.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/872

#### Problem TLDR

Make growing landscape #medium #bfs

#### Intuition

Do BFS from initial points

#### Approach

* next height is always curr + 1
* mark vacant places with `-1` to solve `0` edge case
* fill the place when its added to the queue

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun highestPeak(isWater: Array<IntArray>) = isWater.apply {
        val q = ArrayDeque<Pair<Int, Int>>(); var d = listOf(-1, 0, 1, 0, -1)
        for ((y, r) in withIndex()) for (x in r.indices)
            if (r[x] > 0) { r[x] = 0; q += y to x } else r[x] = -1
        while (q.size > 0) {
            val (y, x) = q.removeFirst()
            for (i in 0..3) {
                val (y1, x1) = y + d[i] to x + d[i + 1]
                if (getOrNull(y1)?.getOrNull(x1) ?: 0 < 0) {
                    this[y1][x1] = 1 + this[y][x]
                    q += y1 to x1
                }
            }
        }
    }

```
```rust

    pub fn highest_peak(mut is_water: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        let mut q = VecDeque::new();
        for y in 0..is_water.len() { for x in 0..is_water[0].len() {
            if is_water[y][x] > 0 { is_water[y][x] = 0; q.push_back((y, x)) }
            else { is_water[y][x] = -1 }
        }}
        while let Some((y, x)) = q.pop_front() {
            for (y1, x1) in [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)] {
                if (y1.min(x1) >= 0 && y1 < is_water.len() && x1 < is_water[0].len()) {
                    if is_water[y1][x1] >= 0 { continue }
                    is_water[y1][x1] = 1 + is_water[y][x];
                    q.push_back((y1, x1))
                }
            }
        }; is_water
    }

```
```c++

    vector<vector<int>> highestPeak(vector<vector<int>>& w) {
        queue<pair<int, int>> q;
        int d[] = {1, 0, -1, 0, 1}, m = size(w), n = size(w[0]);
        for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j)
            if (w[i][j]) w[i][j] = 0, q.push({i, j}); else w[i][j] = -1;
        while (size(q)) {
            auto [y, x] = q.front(); q.pop();
            for (int i = 0; i < 4; ++i)
                if (int y1 = y + d[i], x1 = x + d[i + 1];
                    min(y1, x1) >= 0 && y1 < m && x1 < n && w[y1][x1] < 0)
                    w[y1][x1] = 1 + w[y][x], q.push({y1, x1});
        } return w;
    }

```

