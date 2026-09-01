---
layout: leetcode-entry
title: "3568. Minimum Moves to Clean the Classroom"
permalink: "/leetcode/problem/2026-09-01-3568-minimum-moves-to-clean-the-classroom/"
leetcode_ui: true
entry_slug: "2026-09-01-3568-minimum-moves-to-clean-the-classroom"
---

[3568. Minimum Moves to Clean the Classroom](https://leetcode.com/problems/minimum-moves-to-clean-the-classroom/solutions/8494895/kotlin-by-samoylenkodmitry-nzmq/) medium
[substack](https://dmitriisamoilenko.substack.com/p/01092026-3568-minimum-moves-to-clean?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/knnmmNIP5b0)

https://dmitrysamoylenko.com/leetcode/

![01.09.2026.webp](/assets/leetcode_daily_images/01.09.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1469

#### Problem TLDR

Collect Ls on grid, limited energy E

#### Intuition

Full BFS with visited set tracking for each path. Prune by visited energy per cell: update only improved situation with bigger energy for the same visited set.

#### Approach

* time complexity: each unique state (y,x,m) bounded by energy range 1..E, m = 2^L

#### Complexity

- Time complexity:
$$O(nme2^L)$$

- Space complexity:
$$O(nm2^L)$$

#### Code

```kotlin
    fun minMoves(g: Array<String>, E: Int): Int {
        var sy = 0; var sx = 0; var L = 0; var steps = -1
        val id = Array(g.size) { IntArray(g[0].length) { -1 } }
        for (r in g.indices) for (c in g[0].indices) {
            if (g[r][c] == 'S') { sy = r; sx = c }
            if (g[r][c] == 'L') id[r][c] = L++
        }
        val target = (1 shl L) - 1; if (target == 0) return 0
        val best = Array(g.size) { Array(g[0].length) { IntArray(1 shl L) } }
        val q = ArrayDeque(setOf(intArrayOf(sy, sx, E, 0)))
        while (q.size>0 && ++steps >=0)
            for (i in 1..q.size) {
                val (y, x, e, m) = q.removeFirst()
                for ((dy, dx) in arrayOf(0 to 1, 0 to -1, 1 to 0, -1 to 0)) {
                    val ny = y + dy; val nx = x + dx
                    if (ny in g.indices && nx in g[0].indices && g[ny][nx] != 'X') {
                        val ne = if (g[ny][nx] == 'R') E else e - 1
                        val nm = if (id[ny][nx] >= 0) m or (1 shl id[ny][nx]) else m
                        if (nm == target) return steps + 1
                        if (ne > best[ny][nx][nm]) {
                            best[ny][nx][nm] = ne
                            q.add(intArrayOf(ny, nx, ne, nm))
                        }
                    }
                }
            }
        return -1
    }
```
```rust

```

