---
layout: leetcode-entry
title: "1568. Minimum Number of Days to Disconnect Island"
permalink: "/leetcode/problem/2024-08-11-1568-minimum-number-of-days-to-disconnect-island/"
leetcode_ui: true
entry_slug: "2024-08-11-1568-minimum-number-of-days-to-disconnect-island"
---

[1568. Minimum Number of Days to Disconnect Island](https://leetcode.com/problems/minimum-number-of-days-to-disconnect-island/description/) hard
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/11082024-1568-minimum-number-of-days?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11082024-1568-minimum-number-of-days?r=2bam17&utm_campaign=post&utm_medium=web)
[youtube](https://youtu.be/zx9Hi0rY_Qg)
![1.webp](/assets/leetcode_daily_images/58756713.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/699

#### Problem TLDR

Min changes 1 to 0 to disconnect islands #hard #union-find

#### Intuition

Start with implementing brute force Depth-First Search with backtracking: try to change each 1 to 0 and check if it's disconnected. Use Union-Find to check connected components.

The final solution is just a magic trick: two flips will disconnect every possible case in a 2D grid just by cutting a corner.

I think this problem can be optimized down to a single-pass check for different 1x3, 2x3 and 3x3 patterns to find cases with single flip. Otherwise, it is 0 or 2.

#### Approach

* minor optimization is to consider only the `ones`

#### Complexity

- Time complexity:
$$O((nm)^2)$$

- Space complexity:
$$O(nm)$$

#### Code

```Kotlin

    fun minDays(grid: Array<IntArray>): Int {
        val w = grid[0].size; val h = grid.size; var e = -1; var c = 0
        val ones = (0..<w * h).filter { grid[it / w][it % w] > 0 }
        val uf = IntArray(w * h) { it }
        fun find(a: Int): Int { while (uf[a] != uf[uf[a]]) uf[a] = uf[uf[a]]; return uf[a] }
        fun union(a: Int, b: Int) { if (find(a) != find(b)) { c--; uf[uf[a]] = uf[b] }}
        fun isDisconnected(): Boolean {
            for (i in ones) uf[i] = i; c = 0
            for (i in ones) if (i != e) { c++
                if (i % w > 0 && grid[i / w][i % w - 1] > 0 && i - 1 != e) union(i, i - 1)
                if (i / w > 0 && grid[i / w - 1][i % w] > 0 && i - w != e) union(i, i - w)}
            return c != 1
        }
        return if (isDisconnected()) 0 else if (ones.any { e = it; isDisconnected() }) 1 else 2
    }
```

```rust

    pub fn min_days(grid: Vec<Vec<i32>>) -> i32 {
        let (w, h) = (grid[0].len(), grid.len());
        let ones: Vec<usize> = (0..w * h).filter(|&i| grid[i / w][i % w] > 0).collect();
        fn is_disconnected(grid: &[Vec<i32>], e: usize, ones: &[usize], w: usize, h: usize) -> bool {
            let (mut uf, mut c) = ((0..w * h).collect::<Vec<_>>(), 0);
            fn find(uf: &mut Vec<usize>, x: usize) -> usize {
                while uf[x] != uf[uf[x]] { uf[x] = uf[uf[x]] }; uf[x] }
            for &i in ones.iter() { if i != e { c += 1;
                let mut union = |b: usize| {
                    let mut a = find(&mut uf, i); if a != find(&mut uf, b) { uf[a] = uf[b]; c -= 1 }};
                if i % w > 0 && grid[i / w][i % w - 1] > 0 && i - 1 != e { union(i - 1) }
                if i / w > 0 && grid[i / w - 1][i % w] > 0 && i - w != e { union(i - w) }
            }}; c != 1
        }
        if is_disconnected(&grid, usize::MAX, &ones, w, h) { 0 }
        else if ones.iter().any(|&i| is_disconnected(&grid, i, &ones, w, h)) { 1 } else { 2 }
    }

```

