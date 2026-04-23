---
layout: leetcode-entry
title: "407. Trapping Rain Water II"
permalink: "/leetcode/problem/2025-10-03-407-trapping-rain-water-ii/"
leetcode_ui: true
entry_slug: "2025-10-03-407-trapping-rain-water-ii"
---

[407. Trapping Rain Water II](https://leetcode.com/problems/trapping-rain-water-ii/description) hard
[blog post](https://leetcode.com/problems/trapping-rain-water-ii/solutions/7245035/kotlin-rust-by-samoylenkodmitry-shu9/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/03102025-407-trapping-rain-water?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/x6NSzMUL--Q)

![1.webp](/assets/leetcode_daily_images/6f236dc2.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1131

#### Problem TLDR

Fill water in 3D #hard #dfs #sorting

#### Intuition

I solved it not optimally in 50 minutes O(n^2) (accepted).
Go layer-by-layer, DFS in each layer and find the `min` value of greater cells.

The optimal solution: go layer-by-layer, advance just single step, put back into queue with new height value `min(lvl, next)`.

#### Approach

* use priority_queue
* use visited set or modify the grid

#### Complexity

- Time complexity:
$$O(n^2)$$ or O(nlog(n))

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 1968ms
    fun trapRainWater(h: Array<IntArray>): Int {
        val w = h[0].size;
        val q = PriorityQueue<Int>(compareBy{h[it/w][it%w]}); q += (0..<w*h.size)
        var lvl = 0; var res = 0; var curr = 0; val visited = HashSet<Int>()
        while (q.size > 0) {
            val yx = q.poll(); val (y, x) = yx/w to yx%w; val clvl = h[y][x]
            if (clvl > lvl) { res += curr; curr = 0; lvl = clvl; visited.clear() }
            var min = Int.MAX_VALUE
            fun dfs(y: Int, x: Int): Int {
                if (y<0||x<0||y>h.size-1||x>w-1) { min = 0; return@dfs 0 }
                if (h[y][x] > clvl) {min = min(min, h[y][x]);return@dfs 0}
                if (!visited.add(y*w+x)) return@dfs 0
                return@dfs 1 + dfs(y-1,x) + dfs(y+1,x) + dfs(y,x-1) + dfs(y,x+1)
            }
            curr += max(0, dfs(y,x)*(min-clvl))
        }
        return res
    }

```
```rust

// 7ms
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

