---
layout: leetcode-entry
title: "2812. Find the Safest Path in a Grid"
permalink: "/leetcode/problem/2024-05-15-2812-find-the-safest-path-in-a-grid/"
leetcode_ui: true
entry_slug: "2024-05-15-2812-find-the-safest-path-in-a-grid"
---

[2812. Find the Safest Path in a Grid](https://leetcode.com/problems/find-the-safest-path-in-a-grid/description/) medium
[blog post](https://leetcode.com/problems/find-the-safest-path-in-a-grid/solutions/5159755/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/15052024-2812-find-the-safest-path?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/imflEAHTcRo)
![2024-05-15_09-43.webp](/assets/leetcode_daily_images/624d6a7a.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/603

#### Problem TLDR

Safest path in a grid with thieves #medium #bfs #heap

#### Intuition

Let's firs build a map, marking each cell with its safety number, this can be done with Breadth-First Search from all thieves:
![2024-05-15_07-58.webp](/assets/leetcode_daily_images/cf3b1c45.webp)
The path finding part is straightforward Dijkstra: choose the most optimal path from the heap, stop on the first arrival.

#### Approach

There are some tricks possible:
* use the grid itself as a visited set: check `0` and mark with negative
* we can avoid some extra work if we start safety with `1`

#### Complexity

- Time complexity:
$$O(nmlog(nm))$$

- Space complexity:
$$O(nm)$$

#### Code

```kotlin

    fun maximumSafenessFactor(grid: List<List<Int>>): Int {
        val g = grid.map { it.toTypedArray() }; val n = g.size
        with(ArrayDeque<Pair<Int, Int>>()) {
            for (y in 0..<n) for(x in 0..<n) if (g[y][x] > 0) add(y to x)
            while (size > 0) {
                val (y, x) = removeFirst(); val step = g[y][x] + 1
                fun a(y: Int, x: Int): Unit =
                    if (x in 0..<n && y in 0..<n && g[y][x] < 1) {
                        add(y to x); g[y][x] = step
                    } else Unit
                a(y - 1, x); a(y, x - 1); a(y + 1, x); a(y, x + 1)
            }
        }
        data class Path(val f: Int, val x: Int, val y: Int)
        return with(PriorityQueue<Path>(compareBy { it.f })) {
            add(Path(-g[0][0], 0, 0))
            while (size > 0) {
                val (f, x, y) = poll()
                fun a(x: Int, y: Int): Unit =
                    if (x in 0..<n && y in 0..<n && g[y][x] > 0) {
                        add(Path(-min(-f, g[y][x]), x, y)); g[y][x] *= -1
                    } else Unit
                if (x == n - 1 && y == n - 1) return -f - 1
                a(x - 1, y); a(x, y - 1); a(x + 1, y); a(x, y + 1)
            }; -1
        }
    }

```
```rust

    pub fn maximum_safeness_factor(mut g: Vec<Vec<i32>>) -> i32 {
        let (n, mut q, mut h) = (g.len(), VecDeque::new(), BinaryHeap::new());
        for y in 0..n { for x in 0..n { if g[y][x] > 0 { q.push_back((y, x) )}}}
        while let Some((y, x)) = q.pop_front() {
            let s = g[y][x] + 1;
            if y > 0 && g[y - 1][x] < 1 { q.push_back((y - 1, x)); g[y - 1][x] = s; }
            if x > 0 && g[y][x - 1] < 1 { q.push_back((y, x - 1)); g[y][x - 1] = s; }
            if y < n - 1 && g[y + 1][x] < 1 { q.push_back((y + 1, x)); g[y + 1][x] = s; }
            if x < n - 1 && g[y][x + 1] < 1 { q.push_back((y, x + 1)); g[y][x + 1] = s; }
        }
        h.push((g[0][0], 0, 0));
        while let Some((f, y, x)) = h.pop() {
            if x == n - 1 && y == n - 1 { return f - 1 }
            if y > 0 && g[y - 1][x] > 0 { h.push((f.min(g[y - 1][x]), y - 1, x)); g[y - 1][x] *= -1; }
            if x > 0 && g[y][x - 1] > 0 { h.push((f.min(g[y][x - 1]), y, x - 1)); g[y][x - 1] *= -1; }
            if y < n - 1 && g[y + 1][x] > 0 { h.push((f.min(g[y + 1][x]), y + 1, x)); g[y + 1][x] *= -1; }
            if x < n - 1 && g[y][x + 1] > 0 { h.push((f.min(g[y][x + 1]), y, x + 1)); g[y][x + 1] *= -1; }
        }; -1
    }

```

