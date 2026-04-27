---
layout: leetcode-entry
title: "1391. Check if There is a Valid Path in a Grid"
permalink: "/leetcode/problem/2026-04-27-1391-check-if-there-is-a-valid-path-in-a-grid/"
leetcode_ui: true
entry_slug: "2026-04-27-1391-check-if-there-is-a-valid-path-in-a-grid"
---

[1391. Check if There is a Valid Path in a Grid](https://leetcode.com/problems/check-if-there-is-a-valid-path-in-a-grid/solutions/8102443/kotlin-rust-by-samoylenkodmitry-ifki/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27042026-1391-check-if-there-is-a?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/HlxNEDSXaz0)

https://dmitrysamoylenko.com/leetcode/

![27.04.2026.webp](/assets/leetcode_daily_images/27.04.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1341

#### Problem TLDR

Path 0 to end on type-cells 2D grid

#### Intuition

The union-find idea: connect all cells according to their types. Check if the target type complements current type.
The single path idea: each cell has a single enter/exit so just check path from the left and from the top of the 0 cell.

#### Approach

* we can use transition matrix
* it can be shortened to strings of six sections each of 4 directions: current direction changes to target direction
* 82=2^1+w^4+2^6, 100=2^2+2^5+2^6, 466=2^1+2^4+2^6 + 2^(2+4)+2^(3+4)+2^(4+4)

#### Complexity

- Time complexity:
$$O(nm or path)$$

- Space complexity:
$$O(nm or 1)$$

#### Code

```kotlin
    fun hasValidPath(g: Array<IntArray>): Boolean {
        val w = g[0].size; val u = IntArray(g.size * w) { it }
        fun f(x: Int): Int = if (x == u[x]) x else {u[x]=f(u[x]);u[x]}
        for (i in u.indices) { val x = i%w; val y = i/w
            if (x+1<w&&82 shr g[y][x]and 1>0&&g[y][x+1]%2>0) u[f(i)] = f(i+1)
            if (y+1<g.size&&g[y][x]in 2..4&&100 shr g[y+1][x]and 1>0) u[f(i)]=f(i+w)
        }
        return f(0) == f(u.size - 1)
    }
```
```rust
    pub fn has_valid_path(g: Vec<Vec<i32>>) -> bool {
        (0..2).any(|i| 466 >> g[0][0] + i * 4 & 1 > 0 &&
            successors(Some((0, 0, i)), |&(y, x, d)| {
                let (x,y) = ((x as i32+(1-d)%2)as usize, (y as i32+(2-d)%2)as usize);
                let d = b"....0.2..1.31..2..1032...03."[(*g.get(y)?.get(x)?*4+d)as usize]as i32-48;
                (d >= 0 && x | y > 0).then_some((y, x, d))
            }).any(|(y, x, _)| y == g.len() - 1 && x == g[0].len() - 1)
        )
    }
```

