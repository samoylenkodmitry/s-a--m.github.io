---
layout: leetcode-entry
title: "959. Regions Cut By Slashes"
permalink: "/leetcode/problem/2024-08-10-959-regions-cut-by-slashes/"
leetcode_ui: true
entry_slug: "2024-08-10-959-regions-cut-by-slashes"
---

[959. Regions Cut By Slashes](https://leetcode.com/problems/regions-cut-by-slashes) medium
[blog post](https://leetcode.com/problems/regions-cut-by-slashes/solutions/5615947/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10082024-959-regions-cut-by-slashes?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/UHEK2hSWhpg)
![1.webp](/assets/leetcode_daily_images/2eb066ac.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/698

#### Problem TLDR

Count islands divided by '\' and '/' in 2D matrix #medium #union-find

#### Intuition

Let's divide each cell into four parts: top, right, bottom and left.
![2024-08-10_10-05.png](/assets/leetcode_daily_images/10794669.webp)
Assign a number for each subcell: 0, 1, 2 and 3.
![2024-08-10_10-25.png](/assets/leetcode_daily_images/a959f1f3.webp)
Now, connect cells that are not divided by symbols `/` or `\` and count how many connected components there are. Union-Find is a perfect helper for this task.

#### Approach

Count how many unique roots are left or just decrease the counter when each new connection happens.

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(nm)$$

#### Code

```kotlin

    fun regionsBySlashes(grid: Array<String>): Int {
        val uf = IntArray(grid.size * grid[0].length * 4) { it }; var g = uf.size
        fun find(a: Int): Int { var x = a; while (x != uf[x]) x = uf[x]; uf[a] = x; return x }
        fun union(a: Int, b: Int) { if (find(a) != find(b)) g--; uf[find(a)] = find(b) }
        for ((y, s) in grid.withIndex()) for ((x, c) in s.withIndex()) {
            val k = { d: Int -> y * grid[0].length * 4 + x * 4 + d }
            if (c == '\\') { union(k(0), k(1)); union(k(2), k(3)) }
            else if (c == '/') { union(k(1), k(2)); union(k(0), k(3)) }
            else { union(k(0), k(1)); union(k(1), k(2)); union(k(2), k(3)) }
            if (x > 0) union(k(1) - 4, k(3))
            if (y > 0) union(k(2) - 4 * grid[0].length, k(0))
        }
        return g
    }

```
```rust

    pub fn regions_by_slashes(grid: Vec<String>) -> i32 {
        let mut uf: Vec<_> = (0..grid.len() * grid[0].len() * 4).collect();
        fn find(uf: &mut Vec<usize>, a: usize) -> usize {
            let mut x = a; while x != uf[x] { x = uf[x] }; uf[a] = x; x }
        for (y, s) in grid.iter().enumerate() { for (x, c) in s.chars().enumerate() {
            let k = |d| y * grid[0].len() * 4 + x * 4 + d;
            let mut u = |a, b| { let f = find(&mut uf, a); uf[f] = find(&mut uf, b) };
            if c == '\\' { u(k(0), k(1)); u(k(2), k(3)) }
            else if c == '/' { u(k(1), k(2)); u(k(0), k(3)) }
            else { u(k(0), k(1)); u(k(1), k(2)); u(k(2), k(3)) }
            if x > 0 { u(k(3), k(1) - 4) }
            if y > 0 { u(k(0), k(2) - 4 * grid[0].len()) }
        }}
        (0..uf.len()).map(|x| find(&mut uf, x)).collect::<HashSet<_>>().len() as i32
    }

```

