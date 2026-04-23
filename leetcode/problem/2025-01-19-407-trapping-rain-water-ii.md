---
layout: leetcode-entry
title: "407. Trapping Rain Water II"
permalink: "/leetcode/problem/2025-01-19-407-trapping-rain-water-ii/"
leetcode_ui: true
entry_slug: "2025-01-19-407-trapping-rain-water-ii"
---

[407. Trapping Rain Water II](https://leetcode.com/problems/trapping-rain-water-ii/description/) hard
[blog post](https://leetcode.com/problems/trapping-rain-water-ii/solutions/6302411/kotlin-rust-by-samoylenkodmitry-o2pd/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/19012025-407-trapping-rain-water?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/_hGmmCb8EOk)
![1.webp](/assets/leetcode_daily_images/9b7985a3.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/869

#### Problem TLDR

Trap the water in 2D height matrix #hard #bfs

#### Intuition

Didn't solve this myself in 2 hours.

My naive approach was the brute-force (not accepted, but correct): go layer by layer increasing height, and calculate area with BFS less than current height, track min height difference.

The optimal solution: go from outside with BFS and add height difference, append to the Heap adjacents making them at least current height. Imagine water filling everything at the level of the current `min`.

#### Approach

* spending 2 hours on a wrong idea is ok

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(nm)$$

#### Code

```kotlin

    fun trapRainWater(heightMap: Array<IntArray>): Int {
        val m = heightMap.size - 1; val n = heightMap[0].size - 1; var res = 0
        val q = PriorityQueue<List<Int>>(compareBy { it[0] })
        for (y in 0..m) for (x in 0..n) if (min(x, y) < 1 || y == m || x == n)
            q += listOf(heightMap[y][x], y, x)
        while (q.size > 0) {
            val (min, y, x) = q.poll(); heightMap[y][x] = -1
            for ((y1, x1) in listOf(y to x - 1, y - 1 to x, y to x + 1, y + 1 to x))
                if (y1 in 0..m && x1 in 0..n && heightMap[y1][x1] >= 0) {
                    q += listOf(max(min, heightMap[y1][x1]), y1, x1)
                    res += max(0, min - heightMap[y1][x1]); heightMap[y1][x1] = -1
                }
        }
        return res
    }

```
```rust

    pub fn trap_rain_water(mut height_map: Vec<Vec<i32>>) -> i32 {
        let (m, n, mut r) = (height_map.len(), height_map[0].len(), 0);
        let mut q = BinaryHeap::new();
        for y in 0..m { for x in 0..n { if (y.min(x) < 1 || y == m - 1 || x == n - 1) {
            q.push((-height_map[y][x], y, x)) }}}
        while let Some((min, y, x)) = q.pop() {
            height_map[y][x] = -1; let min = -min;
            for (y1, x1) in [(y, x - 1), (y - 1, x), (y, x + 1), (y + 1, x)] {
                if (0..m).contains(&y1) && (0..n).contains(&x1) && height_map[y1][x1] >= 0 {
                    q.push((-min.max(height_map[y1][x1]), y1, x1));
                    r += 0.max(min - height_map[y1][x1]); height_map[y1][x1] = -1
                }}
        }; r
    }

```
```c++

    int trapRainWater(vector<vector<int>>& g) {
        priority_queue<array<int,3>, vector<array<int,3>>, greater<>> q;
        int m = size(g), n = size(g[0]), r = 0, d[] = {0, 1, 0, -1, 0};
        for (int i = 0; i < m * n; ++i) if (i < n || i >= n * (m - 1) || i % n < 1 || i % n == n - 1)
            q.push({g[i / n][i % n], i / n, i % n });
        while (size(q)) {
            auto [v, y, x] = q.top(); q.pop(); g[y][x] = -1;
            for (int i = 0; i < 4; ++i)
                if (int y1 = y + d[i], x1 = x + d[i + 1]; min(y1, x1) >= 0 && y1 < m && x1 < n && g[y1][x1] >= 0)
                    q.push({max(v, g[y1][x1]), y1, x1}), r += max(0, v - g[y1][x1]), g[y1][x1] = -1;
        } return r;
    }

```

