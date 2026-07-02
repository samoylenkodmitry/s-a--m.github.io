---
layout: leetcode-entry
title: "3286. Find a Safe Walk Through a Grid"
permalink: "/leetcode/problem/2026-07-02-3286-find-a-safe-walk-through-a-grid/"
leetcode_ui: true
entry_slug: "2026-07-02-3286-find-a-safe-walk-through-a-grid"
---

[3286. Find a Safe Walk Through a Grid](https://leetcode.com/problems/find-a-safe-walk-through-a-grid/solutions/8371360/kotlin-rust-by-samoylenkodmitry-y485/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/02072026-3286-find-a-safe-walk-through?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/oCWzK0HLZ2g)

https://dmitrysamoylenko.com/leetcode/

![02.07.2026.webp](/assets/leetcode_daily_images/02.07.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1408

#### Problem TLDR

Path to bottom-right with health points

#### Intuition

* use BFS, put health with coordinates, prioritize the max health first with PriorityQueue
* another way is 0-1 BFS: explore free cells first by putting the in front of the queue
* track the max health for each visited cell to allow for re-enter

#### Approach

* Rust: use !0 instead of -1 as dx/dy

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin
    fun findSafeWalk(g: List<List<Int>>, h: Int): Boolean {
        val q=ArrayDeque(setOf(0 to 0));val v=Array(g.size){IntArray(g[0].size)}; v[0][0]=h-g[0][0]
        while (q.size>0) {
            val (x, y) = q.removeFirst(); if (y == g.size - 1 && x == g[0].size - 1) return true
            for ((X, Y) in setOf(x - 1 to y, x + 1 to y, x to y - 1, x to y + 1))
                if (Y in g.indices && X in g[0].indices && v[y][x] - g[Y][X] > v[Y][X]) {
                    v[Y][X] = v[y][x]-g[Y][X]
                    if (g[Y][X] == 0) q.addFirst(X to Y) else q += X to Y
                }
        }
        return false
    }
```
```rust
    pub fn find_safe_walk(g: Vec<Vec<i32>>, h: i32) -> bool {
        let (R, C) = (g.len(), g[0].len());
        let mut v = vec![vec![0; C]; R]; v[0][0] = h - g[0][0];
        let mut q = VecDeque::from([(0, 0, v[0][0])]);
        while let Some((x, y, h)) = q.pop_front() {
            if h > 0 && x == R - 1 && y == C - 1 { return true }
            for (dx, dy) in [(0,1), (1,0), (0,!0), (!0,0)] {
                let (X, Y) = (x.wrapping_add(dx), y.wrapping_add(dy));
                if X < R && Y < C && h - g[X][Y] > v[X][Y] {
                    v[X][Y] = h - g[X][Y]; let a = (X, Y, v[X][Y]);
                    if g[X][Y] == 0 { q.push_front(a) } else { q.push_back(a) }
                }
            }
        } false
    }
```

